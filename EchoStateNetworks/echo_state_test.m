addpath('../NeuralNetwork/')

% Network - Input layer size
net_in_sz = 1;

% Network - Middle layer size 
net_middle_sz = 80;

% Network - Output layer size
net_out_sz = 1;

% Creating the weigths to the 
% the non feedback echo state network
input_par.sz = [net_in_sz net_middle_sz];
input_par.range = 1;
input_par.sparseness = 1;

feedback_par.sz = [net_middle_sz net_middle_sz];
feedback_par.range = 1;
feedback_par.alpha = 0.98;
feedback_par.sparseness = 1;

[~, ~, Weigths] = generate_echo_state_weigths(input_par, feedback_par);

% Setup training network
clear nn
nn.v = [ones(1, net_middle_sz); Weigths];
nn.b = 0;
nn.func = @tanh;
nn = neuro_net_init(nn);

% Training the network to learn a sin function
% Same target to different inputs
train = linspace(0, 4*pi, 300);
target = sin(train);

reg_factor = 0;
nn.w = calc_esn_weigths(train, target, reg_factor, nn);

output = neural_net_echo_states(train, nn);

plot(output)