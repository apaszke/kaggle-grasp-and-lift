local LSTMSampler = torch.class('LSTMSampler')

function LSTMSampler:__init()
end

function LSTMSampler:load_model(checkpoint, opt)
  self.rnn = checkpoint.protos.rnn
  self.rnn:evaluate() -- put in eval mode so that dropout works properly
  print('creating an LSTM...')
  local num_layers = checkpoint.opt.num_layers
  self.init_state = {}
  for L = 1,num_layers do
      -- c and h for all layers
      local h_init = torch.zeros(1, checkpoint.opt.rnn_size)
      if opt.gpuid >= 0 then h_init = h_init:cuda() end
      table.insert(self.init_state, h_init:clone())
      table.insert(self.init_state, h_init:clone())
  end
  self.state_size = #self.init_state
  self.prediction = torch.zeros(6)
  if opt.gpuid >= 0 then self.prediction = self.prediction:cuda() end
  self.current_state = clone_list(self.init_state)
end

function LSTMSampler:prepare_file(out_file)
  return 0
end

function LSTMSampler:predict(t, data_tensor)
  local lst = self.rnn:forward{data_tensor[t]:view(1, -1), unpack(self.current_state)}
  self.current_state = {}
  for i = 1,self.state_size do table.insert(self.current_state, lst[i]) end
  return lst[#lst]
end
