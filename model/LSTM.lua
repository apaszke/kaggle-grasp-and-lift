require 'nn'
require 'nngraph'

local LSTM = {}
-- n is layer count
function LSTM.lstm(input_size, output_size, rnn_size, n, dropout)
  dropout = dropout or 0

  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()():annotate({ name = "Input" })) -- x
  for L = 1,n do
    table.insert(inputs, nn.Identity()():annotate({ name = "Previous cell at layer " .. L })) -- prev_c[L]
    table.insert(inputs, nn.Identity()():annotate({ name = "Previous output at layer " .. L })) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  -- loop through the layers
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+1]
    local prev_c = inputs[L*2]
    -- the input to this layer
    if L == 1 then
    --   x = nn.Linear(input_size, rnn_size)(inputs[1])
    --   x = nn.ReLU()(x)
    --   input_size_L = rnn_size
    x = inputs[1]
    input_size_L = input_size
    else
      x = outputs[(L-1)*2]
      if dropout > 0 then x = nn.Dropout(dropout)(x):annotate({ name = "Dropout in layer: " .. L }) end -- apply dropout, if any
      input_size_L = rnn_size
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x):annotate({ name = "i2h in layer: " .. L })
    local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h):annotate({ name = "h2h in layer: " .. L })
    -- local c2h = nn.Linear(rnn_size, 3 * rnn_size)(prev_c)
    local all_input_sums = nn.CAddTable()({i2h, h2h}):annotate({ name = "i2h and h2h sum in layer: " .. L })
    -- decode the gates
    local sigmoid_chunk = nn.Narrow(2, 1, 3 * rnn_size)(all_input_sums):annotate({ name = "sigmoid narrow in layer: " .. L })
    -- sigmoid_chunk = nn.CAddTable()({c2h, sigmoid_chunk})
    sigmoid_chunk = nn.Sigmoid()(sigmoid_chunk):annotate({ name = "sigmoid in layer: " .. L })
    local in_gate = nn.Narrow(2, 1, rnn_size)(sigmoid_chunk):annotate({ name = "in gate in layer: " .. L })
    local forget_gate = nn.Narrow(2, rnn_size + 1, rnn_size)(sigmoid_chunk):annotate({ name = "forget gate in layer: " .. L })
    local out_gate = nn.Narrow(2, 2 * rnn_size + 1, rnn_size)(sigmoid_chunk):annotate({ name = "out gate in layer: " .. L })
    -- decode the write inputs
    local in_transform = nn.Narrow(2, 3 * rnn_size + 1, rnn_size)(all_input_sums):annotate({ name = "in transform in layer: " .. L })
    in_transform = nn.Tanh()(in_transform):annotate({ name = "tanh in transform in layer: " .. L })
    -- perform the LSTM update
    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}):annotate({ name = "forget mul in layer: " .. L }),
        nn.CMulTable()({in_gate,     in_transform}):annotate({ name = "in gate mul in layer: " .. L })
      }):annotate({ name = "sum c in layer: " .. L})
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)}):annotate({ name = "output in layer: " .. L })

    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local proj = nn.Linear(rnn_size, output_size)(top_h)
  local out_nonlinearity = nn.Sigmoid()(proj)
  table.insert(outputs, out_nonlinearity)

  return nn.gModule(inputs, outputs)
end

return LSTM
