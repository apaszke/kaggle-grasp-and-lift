require 'nn'

cnn = nn.Sequential()
cnn:add( nn.TemporalConvolution(32, 50, 30) )
cnn:add( nn.ReLU() )
cnn:add( nn.TemporalConvolution(50, 100, 30) )
cnn:add( nn.ReLU() )
cnn:add( nn.TemporalConvolution(100, 300, 242) )
cnn:add( nn.Tanh() )
cnn:add( nn.TemporalConvolution(300, 6, 1))
cnn:add( nn.Sigmoid() )
