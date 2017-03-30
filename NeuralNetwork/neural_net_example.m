% Example: Neural net 
% - 2 inputs
% - 3 neurons in the middle layer
% - activation function: linear function
% - 2 outputs
% - 10 equal samples


% Each sample corresponds to a collumn in the input vector and
% each line corresponds to a different input
in = [1 1 1 1; 2 0 1 3];

% Middle layer weigths 
% Fist line - bias Weights 
% Second line - first input weigths
% Third line - second input weights
nn.v = [1 1 1; 1 2 3; 3 2 1];

% Output layers
% Each line corresponds to the output weights 
% for a different output
% Each collumn corresponds to the weights coming from a different
% neuron of the middle layer
nn.w = [1 2 1 1; 1 1 1 2];

% Activation function 
nn.func = @(x) 0.5*x;
% Bias (equal to all neurons)
nn.b = 1;

% 
nn = neuro_net_init(nn);

% Calculating neuro network output
[out, mid_layer_out] = neural_nete(in, nn);

% Testing neuro network output
assert(all(abs(out(1, 1) - 15.5) < 0.01), 'Wrong output')
assert(all(abs(out(2, 1) - 14.5) < 0.01), 'Wrong output')
assert(all(abs(out(1, 2) - 6.5) < 0.01), 'Wrong output')
assert(all(abs(out(2, 2) - 7.5) < 0.01), 'Wrong output')
