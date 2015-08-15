require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'xlua'

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
cmd:option('-data_dir','data/filtered','directory containing test files')
cmd:text()

lfs.rmdir('tmp')
lfs.mkdir('tmp')
-- parse input params
opt = cmd:parse(arg)

-- check that cunn/cutorch are installed if user wants to use the GPU
if opt.gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

torch.manualSeed(opt.seed)

-- load the model checkpoint
if not lfs.attributes(opt.model, 'mode') then
    print('Error: File ' .. opt.model .. ' does not exist. Are you sure you didn\'t forget to prepend cv/ ?')
end
checkpoint = torch.load(opt.model)
protos = checkpoint.protos
protos.rnn:evaluate() -- put in eval mode so that dropout works properly

-- initialize the rnn state to all zeros
print('creating an LSTM...')
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


prediction = torch.zeros(6)
if opt.gpuid >= 0 then prediction = prediction:cuda() end

-- start sampling/argmaxing

for file in lfs.dir(opt.data_dir) do
    if file:find('data.csv.val') then
        print(file)
        local data_table = {}

        local data_fh = io.open(path.join(opt.data_dir, file))
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

        local out_file = io.open('tmp/' .. file, 'w')

        for t = 1, num_samples do
            if t % 1000 == 0 then
              xlua.progress(t, num_samples)
            end

            local lst = protos.rnn:forward{data_tensor[t]:view(1, -1), unpack(current_state)}
            current_state = {}
            for i = 1,state_size do table.insert(current_state, lst[i]) end
            prediction = lst[#lst]
            for i = 1,prediction:size(2) do
                if i > 1 then
                    out_file:write(',')
                end
                out_file:write(string.format('%.5f', prediction[1][i]))
            end
            out_file:write('\n')
        end

        out_file:close()

        print("")

    end
end
