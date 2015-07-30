
--[[

This file trains a character-level multi-layer RNN on text data

Code is based on implementation in
https://github.com/oxford-cs-ml-2015/practical6
but modified to have multi-layer support, GPU support, as well as
many other common model/optimization bells and whistles.
The practical6 code is in turn based on
https://github.com/wojciechz/learning_to_execute
which is turn based on other stuff in Torch, etc... (long lineage)

]]--

-- s[1][{{}, 1, {}}]

local print_orig = print

-- function print(str)
--     print_orig(os.date('[%H:%M:%S] ') .. tostring(str))
-- end

function printRed(str)
    print('\27[0;31m' .. tostring(str) .. '\27[m')
end

function printGreen(str)
    print('\27[0;32m' .. tostring(str) .. '\27[m')
end

function printYellow(str)
    print('\27[0;33m' .. tostring(str) .. '\27[m')
end

function printBlue(str)
    print('\27[0;34m' .. tostring(str) .. '\27[m')
end

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'gnuplot'

require 'util.misc'
local model_utils = require 'util.model_utils'
local LSTM = require 'model.LSTM'
local EEGMinibatchLoader = require 'util.EEGMinibatchLoader'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a character-level language model')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir','data/filtered','data directory')
cmd:option('-prepro_dir','data/preprocessed','preprocessed data directory')
-- model params
cmd:option('-rnn_size', 128, 'size of LSTM internal state')
cmd:option('-num_layers', 2, 'number of layers in the LSTM')
-- optimization
cmd:option('-optim_algo','rmsprop','optimization algorithm')
cmd:option('-learning_rate',2e-3,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-dropout',0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-seq_length',250,'number of timesteps to unroll for')
cmd:option('-batch_size',50,'number of sequences to train on in parallel')
cmd:option('-max_epochs',50,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-test_files',2,'numer of files that go into test set')
cmd:option('-val_files',3,'numer of files that go into validation set')
            -- remaining files will be used for training
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
-- bookkeeping
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-print_every',1,'how many steps/minibatches between printing out the loss')
cmd:option('-eval_val_every',1000,'every how many iterations should we evaluate on validation data?')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile','lstm','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
-- GPU/CPU
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then printRed('package cunn not found!') end
    if not ok2 then printRed('package cutorch not found!') end
    if ok and ok2 then
        printGreen('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        printYellow('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        printYellow('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        printYellow('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- create the data loader class
local loader = EEGMinibatchLoader.create(opt.data_dir, opt.prepro_dir, opt)

-- make sure output directory exists
if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end

-- define the model: prototypes for one timestep, then clone them in time
local do_random_init = true
local start_iter = 1
if string.len(opt.init_from) > 0 then
    print('loading an LSTM from checkpoint ' .. opt.init_from)
    local checkpoint = torch.load(opt.init_from)
    protos = checkpoint.protos
    -- overwrite model settings based on checkpoint to ensure compatibility
    print('overwriting rnn_size=' .. checkpoint.opt.rnn_size .. ', num_layers=' .. checkpoint.opt.num_layers .. ' based on the checkpoint.')
    opt.rnn_size = checkpoint.opt.rnn_size
    opt.num_layers = checkpoint.opt.num_layers
    start_iter = checkpoint.i
    do_random_init = false

    loader.file_idx = checkpoint.loader.file_idx
    loader.batch_idx = checkpoint.loader.batch_idx
    loader:refresh()
else
    print('creating an LSTM with ' .. opt.rnn_size .. ' units in ' .. opt.num_layers .. ' layers')
    protos = {}
    protos.rnn = LSTM.lstm(loader.input_dim, loader.label_dim, opt.rnn_size, opt.num_layers, opt.dropout) -- TODO: set proper size
    protos.criterion = nn.BCECriterion()
end

-- the initial state of the cell/hidden states
init_state = {}
for L=1,opt.num_layers do
    local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
    if opt.gpuid >=0 then h_init = h_init:cuda() end
    table.insert(init_state, h_init:clone())
    table.insert(init_state, h_init:clone())
end

-- ship the model to the GPU if desired
if opt.gpuid >= 0 then
    for k,v in pairs(protos) do v:cuda() end
end

-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(protos.rnn)

-- initialization
if do_random_init then
    params:uniform(-0.08, 0.08) -- small numbers uniform
end

print('number of parameters in the model: ' .. params:nElement())
-- make a bunch of clones after flattening, as that reallocates memory
clones = {}
for name,proto in pairs(protos) do
    print('cloning ' .. name)
    clones[name] = model_utils.clone_many_times(proto, opt.seq_length) -- NOTE: deleted , not proto.parameters)
end


-- evaluate the loss over an entire split
function eval_split(split_index)
    print('evaluating loss over split index ' .. split_index)

    loader:reset_batch_pointer(split_index) -- move batch iteration pointer for this split to front
    local loss = 0
    local rnn_state = {[0] = init_state}

    -- TODO: dirty hack. will work as long as there are less then 1e6 batches in a file
    function get_batch_id()
        return loader.file_idx[split_index] * 1e6 + loader.batch_idx[split_index]
    end

    -- iterate over batches in the split
    local ct = 0
    local last_batch_id = -1
    while get_batch_id() > last_batch_id do
        last_batch_id = get_batch_id()
        -- fetch a batch
        local x, y = loader:next_batch(split_index)
        if opt.gpuid >= 0 then -- ship the input arrays to GPU
            -- have to convert to float because integers can't be cuda()'d
            x = x:float():cuda()
            y = y:float():cuda()
        end
        -- forward pass
        for t=1,opt.seq_length do
            clones.rnn[t]:evaluate() -- for dropout proper functioning
            local lst = clones.rnn[t]:forward{x[{{}, t, {}}], unpack(rnn_state[t-1])}
            rnn_state[t] = {}
            for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end
            prediction = lst[#lst]
            loss = loss + clones.criterion[t]:forward(prediction, y[{{}, t, {}}])
        end
        -- carry over lstm state
        rnn_state[0] = rnn_state[#rnn_state]
        ct = ct + 1
        if ct % 10 == 0 then
            print(ct .. '...')
        end
    end

    loss = loss / opt.seq_length / ct
    return loss
end

-- do fwd/bwd and return loss, grad_params
local init_state_global = clone_list(init_state)
function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

    ------------------ get minibatch -------------------
    local x, y = loader:next_batch(1)
    if opt.gpuid >= 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x:float():cuda()
        y = y:float():cuda()
    end
    ------------------- forward pass -------------------
    local rnn_state = {[0] = init_state_global}
    local predictions = {}           -- softmax outputs
    local loss = 0
    for t=1,opt.seq_length do
        clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
        local lst = clones.rnn[t]:forward{x[{{}, t, {}}], unpack(rnn_state[t-1])}
        rnn_state[t] = {}
        for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
        predictions[t] = lst[#lst] -- last element is the prediction
        -- if t % 25 == 0 then
        --     local str = ''
        --     for i = 1,6 do
        --         str = str .. string.format('%.2f ', predictions[t][1][i])
        --     end
        --     str = str .. '\n'
        --     for i = 1,6 do
        --         str = str .. string.format('%.2f ', y[{{}, t, {}}][1][i])
        --     end
        --     print(str .. '| ' .. t)
        -- end
        loss = loss + clones.criterion[t]:forward(predictions[t], y[{{}, t, {}}])
    end
    loss = loss / opt.seq_length
    ------------------ backward pass -------------------
    -- initialize gradient at time t to be zeros (there's no influence from future)
    local drnn_state = {[opt.seq_length] = clone_list(init_state, true)} -- true also zeros the clones
    for t=opt.seq_length,1,-1 do
        -- backprop through loss, and softmax/linear
        local doutput_t = clones.criterion[t]:backward(predictions[t], y[{{}, t, {}}])
        table.insert(drnn_state[t], doutput_t)
        local dlst = clones.rnn[t]:backward({x[{{}, t, {}}], unpack(rnn_state[t-1])}, drnn_state[t])
        drnn_state[t-1] = {}
        for k,v in pairs(dlst) do
            if k > 1 then -- k == 1 is gradient on x, which we dont need
                -- note we do k-1 because first item is dembeddings, and then follow the
                -- derivatives of the state, starting at index 2. I know...
                drnn_state[t-1][k-1] = v
            end
        end
    end
    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    -- init_state_global = rnn_state[#rnn_state] -- NOTE: I don't think this needs to be a clone, right?
    -- clip gradient element-wise
    grad_params:div(opt.seq_length)
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    return loss, grad_params
end

function calculate_avg_loss(losses)
    local smoothing = 40
    local sum = 0
    for i = #losses, math.max(1, #losses - smoothing + 1), -1 do
        sum = sum + losses[i]
    end
    return sum / math.min(smoothing, #losses)
end

-- start optimization here
train_losses = {}
train_losses_avg = {}
val_losses = {}
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local iterations = opt.max_epochs * loader.total_samples
local loss0 = nil
for i = start_iter, iterations do
    local epoch = i / loader.total_samples

    local _, loss
    local timer = torch.Timer()
    if opt.optim_algo == 'rmsprop' then
        local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
        _, loss = optim.rmsprop(feval, params, optim_state)
    elseif opt.optim_algo == 'adadelta' then
        local optim_state = {rho = 0.95, eps = 1e-7}
        _, loss = optim.adadelta(feval, params, optim_state)
    end
    local time = timer:time().real

    local train_loss = loss[1] -- the loss is inside a list, pop it
    train_losses[i] = train_loss
    train_losses_avg[i] = calculate_avg_loss(train_losses)

    if i % opt.print_every == 0 then
        local grad_norm = grad_params:norm()
        local param_norm = params:norm()
        print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, param norm = %.2e time/batch = %.2fs",
                i, iterations, epoch, train_loss, grad_norm / param_norm, param_norm, time))
        local ct = 0;
        local xAxis = torch.Tensor(#train_losses_avg):apply(function() ct = ct + 1; return ct; end)
        gnuplot.plot(xAxis, torch.Tensor(train_losses_avg))
        gnuplot.figure(2)
        gnuplot.plot(xAxis:sub(1, #val_losses), torch.Tensor(val_losses))
        gnuplot.figure(1)
    end

    -- exponential learning rate decay
    if i % loader.total_samples == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
        end
    end

    -- every now and then or on last iteration
    if i % opt.eval_val_every == 0 or i == iterations then
        -- evaluate loss on validation data
        local val_loss = eval_split(2) -- 2 = validation
        val_losses[i] = val_loss

        local savefile = string.format('%s/lm_%s_epoch%.4f_%.2f.t7', opt.checkpoint_dir, opt.savefile, val_loss, epoch)
        print('saving checkpoint to ' .. savefile)
        local checkpoint = {}
        checkpoint.protos = protos
        checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.val_loss = val_loss
        checkpoint.val_losses = val_losses
        checkpoint.i = i
        checkpoint.epoch = epoch
        checkpoint.loader = {}
        checkpoint.loader.file_idx = loader.file_idx
        checkpoint.loader.batch_idx = loader.batch_idx
        torch.save(savefile, checkpoint)
    end

    if i % 10 == 0 then collectgarbage() end

    -- handle early stopping if things are going really bad
    if loss[1] ~= loss[1] then
        print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
        break -- halt
    end
    if loss0 == nil then loss0 = loss[1] end
    if loss[1] > loss0 * 3 then
        print('loss is exploding, aborting.')
        break -- halt
    end
end


print 'TRAINING DONE'
