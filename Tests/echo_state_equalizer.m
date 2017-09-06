addpath('../NeuralNetwork/')
addpath('../EchoStateNetworks/')
addpath('../Kernels/')


% Generating linear equalizer data
test_size = 1e4;
train_size = 5e2;

% Equalizer channel
h = [0.5 1];

% Random BPSK channel input
rand('state', 0)
bpsk_input_train = sign(rand(1, train_size) - 0.5);
channel_output_train = filter(h, 1, bpsk_input_train);

bpsk_input_test = sign(rand(1, test_size) - 0.5);
channel_output_test = filter(h, 1, bpsk_input_test);

% Network - Input layer size
net_in_sz = 1;

% Network - Middle layer size 
net_middle_sz = 5;

% Network - Output layer size
net_out_sz = 1;

% Creating the weigths to the 
% the non feedback echo state network
input_par.sz = [net_in_sz net_middle_sz];
input_par.range = 1;
% input_par.sparseness = 1;

feedback_par.sz = [net_middle_sz net_middle_sz];
feedback_par.range = 1;
feedback_par.alpha = 0.95;
% feedback_par.sparseness = 1;

[~, ~, Weigths] = generate_echo_state_weigths(input_par, feedback_par);

% Setup training network
clear nn
nn.v = Weigths;
nn.b = 0;
nn.func = @tanh;
nn = neuro_net_init(nn);

% Training the network to learn a sin function
% Same target to different inputs
train = channel_output_train;
target_train = bpsk_input_train;

test = channel_output_test;
target_test = bpsk_input_test;

reg_factor = 1e-7;

% Removing not used bias term
[~, H_train] = calc_esn_weigths(train, target_train, reg_factor, nn);
[~, H_test] = calc_esn_weigths(test, target_test, reg_factor, nn);

% Removing initial samples from data
init_samples = 50;
H_train = H_train(init_samples:end, 2:end);
H_test = H_test(init_samples:end, 2:end);
target_train = target_train(init_samples:end);
target_test = target_test(init_samples:end);

%kernel_train = poly_kernel(H_train, H_train, 1, 1, 3);
%kernel_test = poly_kernel(H_test, H_train, 1, 1, 3);

%kernel_train = linear_kernel(H_train, H_train);
%kernel_test = linear_kernel(H_test, H_train);

kernel_train = rbf_kernel(H_train, H_train, 1);
kernel_test = rbf_kernel(H_test, H_train, 1);

kernel_coef = (kernel_train + reg_factor*eye(size(kernel_train)))\target_train';

estimated_test_data = kernel_test*kernel_coef;

errors = sum(sign(estimated_test_data) != target_test')/(test_size-init_samples);
