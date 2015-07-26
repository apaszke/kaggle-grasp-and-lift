
local EEGMinibatchLoader = {}
EEGMinibatchLoader.__index = EEGMinibatchLoader

-- split_index is integer: 1 = train, 2 = val, 3 = test

local PREPRO_TABLE_THRESHOLD = 0.7e6

function EEGMinibatchLoader.create(data_dir, prepro_dir, opt)

    local self = {}
    setmetatable(self, EEGMinibatchLoader)

    self.x_prepro_prefix = path.join(prepro_dir, 'data')
    self.y_prepro_prefix = path.join(prepro_dir, 'label')
    self.prepro_dir = prepro_dir
    self.batch_size = opt.batch_size
    self.seq_length = opt.seq_length

    -- fetch file attributes to determine if we need to rerun preprocessing
    local run_prepro = false
    if (not path.exists(self.x_prepro_prefix .. '_train1.t7') or not path.exists(self.y_prepro_prefix .. '_train1.t7')) then
        -- prepro files do not exist, generate them
        print('data1.t7 or label1.t7 doesn\'t exist. Running preprocessing...')
        run_prepro = true
    end

    if run_prepro then
        -- preprocess files and save the tensors
        print('one-time setup: preprocessing input...')
        local data_files, val_files, test_files =
                EEGMinibatchLoader.glob_raw_data_files(data_dir, opt.test_files, opt.val_files)
        print('parsing training data')
        self.total_samples = EEGMinibatchLoader.preprocess(data_files,
                                                           self.x_prepro_prefix .. '_train',
                                                           self.y_prepro_prefix .. '_train',
                                                           opt)
        torch.save(path.join(prepro_dir, 'sample_count.t7'), self.total_samples)
        print('parsing validation data')
        EEGMinibatchLoader.preprocess(val_files,
                                      self.x_prepro_prefix .. '_val',
                                      self.y_prepro_prefix .. '_val',
                                      opt)
        print('parsing test data')
        EEGMinibatchLoader.preprocess(test_files,
                                      self.x_prepro_prefix .. '_test',
                                      self.y_prepro_prefix .. '_test',
                                      opt)
    end

    local train_ct, val_ct, test_ct = self.count_prepro_files(prepro_dir)
    self.file_count = {train_ct, val_ct, test_ct}
    self.data_loaded = {false, false, false}
    self.total_samples = torch.load(path.join(prepro_dir, 'sample_count.t7')) / opt.batch_size / opt.seq_length
    self.batch_idx = {0, 0, 0}
    self.file_idx = {1, 1, 1}

    self:load_file(1, 1)

    print('data load done.')
    collectgarbage()
    return self
end

function EEGMinibatchLoader:load_file(split_index, index)
    if split_index < 1 or split_index > 3 then
        printRed('invalid split index in load_file: ' .. split_index)
        os.exit()
    end
    if self.file_count[split_index] < index then
        printRed('invalid file index in load_file: split=' .. split_index .. ', index=' .. index)
        os.exit()
    end

    local split_names = {'_train', '_val', '_test'}
    local modifier = split_names[split_index]

    local x_path = self.x_prepro_prefix .. modifier .. index .. '.t7'
    local y_path = self.y_prepro_prefix .. modifier .. index .. '.t7'
    if (not path.exists(x_path) or not path.exists(y_path)) then
        printRed('trying to load inexistent files! (' .. x_path .. ', ' .. y_path .. ')')
        os.exit()
    end

    print('loading data part ' .. index .. ' from split ' .. split_index)
    local data = torch.load(x_path)
    local labels = torch.load(y_path)

    -- cut off the end so that it divides evenly
    local len = data:size(1)
    if len % (self.batch_size * self.seq_length) ~= 0 then
        print('cutting off end of data so that the batches/sequences divide evenly')
        local new_len = self.batch_size * self.seq_length
                    * math.floor(len / (self.batch_size * self.seq_length))
        data = data:sub(1, new_len)
        labels = labels:sub(1, new_len)
        printYellow('wasted ' .. (len - new_len) .. ' samples out of ' .. len)
    end

    -- get input and label dimensionality
    self.input_dim = data:size(2)
    self.label_dim = labels:size(2)

    -- (x, y, z) = (batch_nr, sample_nr, feat_nr)
    self.x_batches = data:view(self.batch_size, -1, self.input_dim):split(self.seq_length, 2)
    self.nbatches = #self.x_batches
    self.y_batches = labels:view(self.batch_size, -1, self.label_dim):split(self.seq_length, 2)
    assert(#self.x_batches == #self.y_batches)
    printBlue('loaded ' .. self.nbatches .. ' batches')

    self.data_loaded = {false, false, false}
    self.data_loaded[split_index] = true
    self.file_idx[split_index] = index
    self.batch_idx[split_index] = 0

    return true
end

function EEGMinibatchLoader:reset_batch_pointer(split_index, batch_index, file_index)
    batch_index = batch_index or 0
    file_index = file_index or 1
    self.batch_idx[split_index] = batch_index
    self.file_idx[split_index] = file_index
end

-- TODO: add support for other sets
function EEGMinibatchLoader:refresh()
    local prev_batch_idx = self.batch_idx[1]
    self:load_file(1, self.file_idx[1])
    self.batch_idx[1] = prev_batch_idx
    print('resuming from batch ' .. prev_batch_idx)
end

function EEGMinibatchLoader:next_batch(split_index)
    -- load data
    if not self.data_loaded[split_index] then
        local prev_batch_idx = self.batch_idx[split_index]
        self:load_file(split_index, self.file_idx[split_index])
        self.batch_idx[split_index] = prev_batch_idx
        if prev_batch_idx > 0 then
            print('resuming from batch ' .. prev_batch_idx)
        end
    end

    self.batch_idx[split_index] = self.batch_idx[split_index] + 1
    if self.batch_idx[split_index] > #self.x_batches then
        -- load next file
        local file_idx = self.file_idx[split_index] + 1
        if file_idx > self.file_count[split_index] then
            file_idx = 1 -- wrap around file count
        end
        self:load_file(split_index, file_idx) -- sets new file index
        self.batch_idx[split_index] = 1
    end

    local final_batch_idx = self.batch_idx[split_index]
    return self.x_batches[final_batch_idx], self.y_batches[final_batch_idx]
end

-- *** STATIC methods ***
function EEGMinibatchLoader.count_prepro_files(prepro_dir)
    local train = 0
    local val = 0
    local test = 0

    for file in lfs.dir(prepro_dir) do
        if file:find('data_train') then
            train = train + 1
        elseif file:find('data_val') then
            val = val + 1
        elseif file:find('data_test') then
            test = test + 1
        end

    end

    return train, val, test
end


function EEGMinibatchLoader.glob_raw_data_files(data_dir, num_test, num_val)
    local data = {}
    local val_data = {}
    local test_data = {}
    local current_table = test_data

    local ct = 0
    for file in lfs.dir(data_dir) do
        if file:find('data.csv') then
            table.insert(current_table, path.join(data_dir, file))
            ct = ct + 1
        end

        if ct >= num_test + num_val then
            current_table = data
        elseif ct >= num_test then
            current_table = val_data
        end
    end

    printBlue(string.format('found %d training files, %d validation files and %d test files', #data, #val_data, #test_data))

    return data, val_data, test_data
end

function EEGMinibatchLoader.preprocess(input_files, input_filename, label_filename, opt)
    local data_table = {}
    local label_table = {}
    local timer = torch.Timer()
    local total_samples = 0

    local data_idx = 1

    -- helper function
    function saveTensors()
        total_samples = total_samples + #data_table
        assert(#data_table == #label_table)
        printYellow("Cleaning up...")
        collectgarbage()

        printYellow("Saving tensors...")
        torch.save(input_filename .. data_idx .. '.t7', torch.Tensor(data_table))
        torch.save(label_filename .. data_idx .. '.t7', torch.Tensor(label_table))

        -- empty temporary tables
        data_idx = data_idx + 1
        data_table = {}
        label_table = {}
        collectgarbage()
    end

    for i,data_file in ipairs(input_files) do

        print("[" .. i .. "/" .. #input_files .. "] " .. path.basename(data_file))

        ------------------ data file -------------------
        local data_fh = io.open(data_file)
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

        ----------------- events file ------------------
        local labels_csv = string.gsub(data_file, "data.csv", "events.csv") -- change the file name
        local label_fh = io.open(labels_csv)
        if not label_fh then
            printRed('ERROR. Couldn\'t find events file: ' .. labels_csv)
            os.exit()
        end

        local label_content = label_fh:read('*all'):split('\n')
        label_fh:close()

        -- parse event file
        for i, line in ipairs(label_content) do
            if i > 1 then -- ignore header
                local fields = line:split(',')
                table.remove(fields, 1)
                table.insert(label_table, fields)
            end
        end

        ----------------- save tensors -----------------
        -- save tensors if tables are growing big
        -- for lua's 2GB memory limit 1e6 entries is already big
        if #data_table > PREPRO_TABLE_THRESHOLD then
            saveTensors()
        end
    end


    if #data_table > 0 then
        saveTensors()
    end
    printYellow(string.format("Elapsed: %.2fs", timer:time().real))
    return total_samples
end

return EEGMinibatchLoader
