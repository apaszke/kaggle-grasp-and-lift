local CNNSampler = torch.class('CNNSampler')

function CNNSampler:__init()
end

function CNNSampler:load_model(checkpoint)
  print('creating CNN')
  self.cnn = checkpoint.cnn
  self.cnn:evaluate()
  self.window_len = checkpoint.opt.window_len
end

function CNNSampler:prepare_file(out_file)
  for i = 1, self.window_len do
    out_file:write('0,0,0,0,0,0\n')
  end
  return self.window_len
end

function CNNSampler:predict(t, data_tensor)
  local first_sample = t - self.window_len + 1
  local x = data_tensor:sub(first_sample, t)
  return self.cnn:forward(x)
end
