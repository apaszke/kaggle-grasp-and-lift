
--[[

This file samples characters from a trained model

Code is based on implementation in
https://github.com/oxford-cs-ml-2015/practical6

]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'util.misc'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Sample from a character-level language model')
cmd:text()
cmd:text('Options')
-- required:
cmd:argument('-model','model checkpoint to use for sampling')
-- optional parameters
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-sample',1,' 0 to use max at each timestep, 1 to sample at each timestep')
cmd:option('-temperature',1,'temperature of sampling')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-verbose',1,'set to 0 to ONLY print the sampled text, no diagnostics')
cmd:option('-test_data_dir','data/test','directory containing test files')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

-- gated print: simple utility function wrapping a print
function gprint(str)
    if opt.verbose == 1 then print(str) end
end

-- check that cunn/cutorch are installed if user wants to use the GPU
if opt.gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then gprint('package cunn not found!') end
    if not ok2 then gprint('package cutorch not found!') end
    if ok and ok2 then
        gprint('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        gprint('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

torch.manualSeed(opt.seed)

-- load the model checkpoint
if not lfs.attributes(opt.model, 'mode') then
    gprint('Error: File ' .. opt.model .. ' does not exist. Are you sure you didn\'t forget to prepend cv/ ?')
end
checkpoint = torch.load(opt.model)
protos = checkpoint.protos
protos.rnn:evaluate() -- put in eval mode so that dropout works properly

-- initialize the rnn state to all zeros
gprint('creating an LSTM...')
local init_state
local num_layers = checkpoint.opt.num_layers
init_state = {}
for L = 1,num_layers do
    -- c and h for all layers
    local h_init = torch.zeros(1, checkpoint.opt.rnn_size)
    if opt.gpuid >= 0 then h_init = h_init:cuda() end
    table.insert(init_state, h_init:clone())
    table.insert(init_state, h_init:clone())
end
state_size = #init_state

-- do a few seeded timesteps
-- local seed_text = opt.primetext
-- if string.len(seed_text) > 0 then
--     gprint('seeding with ' .. seed_text)
--     gprint('--------------------------')
--     for c in seed_text:gmatch'.' do
--         prev_char = torch.Tensor{vocab[c]}
--         io.write(ivocab[prev_char[1]])
--         if opt.gpuid >= 0 and opt.opencl == 0 then prev_char = prev_char:cuda() end
--         if opt.gpuid >= 0 and opt.opencl == 1 then prev_char = prev_char:cl() end
--         local lst = protos.rnn:forward{prev_char, unpack(current_state)}
--         -- lst is a list of [state1,state2,..stateN,output]. We want everything but last piece
--         current_state = {}
--         for i=1,state_size do table.insert(current_state, lst[i]) end
--         prediction = lst[#lst] -- last element holds the log probabilities
--     end
-- else
    -- fill with uniform probabilities over characters (? hmm)
    -- gprint('missing seed text, using uniform probability over first character')
    -- gprint('--------------------------')
    prediction = torch.zeros(6)
    if opt.gpuid >= 0 then prediction = prediction:cuda() end
-- end

-- start sampling/argmaxing

for file in lfs.dir(opt.test_data_dir) do
    if file:find('data.csv') then
        print(file)
        local data_table = {}

        local data_fh = io.open(path.join(opt.test_data_dir, file))
        local data_content = data_fh:read('*all'):split('\n')
        data_fh:close()

        -- parse data file
        for i,line in ipairs(data_content) do
            if i > 1 then -- ignore header
                local fields = line:split(',')
                table.remove(fields, 1)
                table.insert(data_table, fields)
            end
        end

        local data_tensor = torch.Tensor(data_table)
        if opt.gpuid >= 0 then data_tensor = data_tensor:cuda() end
        local num_samples = data_tensor:size(1)

        print('read ' .. num_samples .. ' samples')

        current_state = clone_list(init_state)

        local out_file = io.open('test_out', 'w')

        for t = 1, num_samples do
            if t % 1000 == 0 then
                print(t)
            end

            local lst = protos.rnn:forward{data_tensor[t]:view(1, -1), unpack(current_state)}
            current_state = {}
            for i = 1,state_size do table.insert(current_state, lst[i]) end
            prediction = lst[#lst]
            if prediction:max() > 0.3 then
                print(prediction)
            end
            for i = 1,prediction:size(2) do
                if i > 1 then
                    out_file:write(',')
                end
                out_file:write(prediction[1][i])
            end
            out_file:write('\n')
        end

        out_file:close()

        os.exit()
    end
end
