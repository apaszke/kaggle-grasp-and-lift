
require 'torch'
require 'nn'
require 'optim'
require 'lfs'
require 'gnuplot'
require 'util.print'

local MODEL_ID = torch.randn(1)[1]
local EEGMinibatchLoader = require 'util.EEGMinibatchLoader'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a cnn to classify EEG recordings')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir','data/filtered','data directory')
cmd:option('-prepro_dir','data/preprocessed','preprocessed data directory')
-- model prototype
cmd:option('-proto_file', 'cnn/proto/first_cnn.lua', 'file defining network structure')
-- optimization
cmd:option('-optim_algo','rmsprop','optimization algorithm')
cmd:option('-learning_rate',2e-3,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-dropout',0,'dropout for regularization (0 = no dropout)')
cmd:option('-seq_length',800,'batch length')
cmd:option('-batch_size',2,'number of sequences to train on in parallel')
cmd:option('-window_len',300,'cnn window size')
cmd:option('-max_epochs',50,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at this value')
-- checkpoints
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
-- bookkeeping
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-print_every',1,'how many steps/minibatches between printing out the loss')
cmd:option('-eval_val_every',1000,'every how many iterations should we evaluate on validation data?')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile','cnn','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
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
local loader = EEGMinibatchLoader.create(opt)

-- make sure output directory exists
if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end

-- define the model: prototypes for one timestep, then clone them in time
local do_random_init = true
local start_iter = 1
if string.len(opt.init_from) > 0 then
    printRed('Checkpoints aren\'t supported yet!')
    os.exit()
else
    print('creating CNN')
    dofile(opt.proto_file)
    criterion = nn.BCECriterion()
end

-- ship the model to the GPU if desired
if opt.gpuid >= 0 then
    cnn:cuda()
    criterion:cuda()
end

-- evaluate the loss over an entire split
function eval_split(split_index)
    print('evaluating loss over split index ' .. split_index)

    loader:reset_batch_pointer(split_index) -- move batch iteration pointer for this split to front
    local loss = 0

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
        local batch_size = x:size(1)
        local num_steps = x:size(2) - opt.window_len + 1
        local partial_loss = 0
        for first_sample = 1, num_steps do
            local last_sample = first_sample + opt.window_len - 1
            local x_mini = x:sub(1, batch_size, first_sample, last_sample)
            local y_mini = y[{{}, last_sample, {}}]
            partial_loss = partial_loss + criterion:forward(cnn:forward(x_mini), y_mini)
        end
        loss = loss + (partial_loss / num_steps)
        ct = ct + 1
        if ct % 10 == 0 then
            print('Evaluated: ' .. ct .. ' batches')
        end
    end

    loss = loss / ct
    return loss
end

local params, grad_params = cnn:getParameters()
params:uniform(-0.08, 0.08)
local feval = function(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

    local x, y = loader:next_batch(1)
    if opt.gpuid >= 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x:float():cuda()
        y = y:float():cuda()
    end

    local loss = 0
    local batch_size = x:size(1)
    local num_steps = x:size(2) - opt.window_len + 1
    for first_sample = 1, num_steps do
        local last_sample = first_sample + opt.window_len - 1
        local x_mini = x:sub(1, batch_size, first_sample, last_sample)
        local y_mini = y[{{}, last_sample, {}}]
        -- print(cnn:forward(x_mini):size())
        local partial_loss = criterion:forward(cnn:forward(x_mini), y_mini)
        loss = loss + partial_loss
        cnn:backward(x_mini, criterion:backward(cnn.output, y_mini))
        if first_sample == 1 then
            str = '\n'
            for i = 1, 6 do
                str = str .. string.format('%.2f ', cnn.output[1][1][i])
            end
            str = str .. '\n'
            for i = 1, 6 do
                str = str .. string.format('%.2f ', y_mini[1][i])
            end
            print(str)
        end
    end

    grad_params:div(num_steps)
    grad_params:clamp(-5, 5)
    loss = loss / num_steps
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
train_losses = train_losses or {}
train_losses_avg = train_losses_avg or {}
val_losses = val_losses or {}

local optim_fun, optim_state
if opt.optim_algo == 'rmsprop' then
    optim_fun = optim.rmsprop
    optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
elseif opt.optim_algo == 'adadelta' then
    optim_fun = optim.adadelta
    optim_state = {rho = 0.95, eps = 1e-7}
end

local iterations = opt.max_epochs * loader.total_samples
local loss0 = nil
for i = start_iter, iterations do
    local epoch = i / loader.total_samples

    local timer = torch.Timer()
    local _, loss = optim_fun(feval, params, optim_state)
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
        checkpoint.cnn = cnn
        checkpoint.criterion = criterion
        checkpoint.type = "cnn"
        checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.val_loss = val_loss
        checkpoint.val_losses = val_losses
        checkpoint.i = i
        checkpoint.epoch = epoch
        checkpoint.loader = {}
        checkpoint.loader.file_idx = loader.file_idx
        checkpoint.loader.batch_idx = loader.batch_idx
        checkpoint.id = MODEL_ID
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
