addpath('../RadialBasesNetwork')
addpath('../NeuralNetwork')
addpath('../TrainingMethods')
addpath('../BackPropagation')
addpath('../ThirdParty/')
addpath('../LineSearchs/')

%%
% Loading training data
load('channel_training_data.mat')

% Loading channels

% Case 1
h_case1 = [0.9 1];                          % Channel
d_case1 = 0;                                % Delay

% Case 2
h_case2 = [1 0.5 0.2];                      % Channel
d_case2 = 2;                                % Delay

% Neural network structure case_1
clear nn_case1
in_sz = 2;
mid_layer_sz = 16;
out_sz = 1;
nn_case1.b = 1;
nn_case1.v = 1*rand(in_sz+1, mid_layer_sz);
nn_case1.w = 1*rand(1, mid_layer_sz+1);
nn_case1.func = @(x) tanh(x);
nn_case1.diff = @(x) 1 - tanh(x).^2;

nn_case1 = neuro_net_init(nn_case1);

% Neural network structure case_2
clear nn_case2
in_sz = 2;
mid_layer_sz = 10;
out_sz = 1;
nn_case2.b = 1;
nn_case2.v = 1*rand(in_sz+1, mid_layer_sz);
nn_case2.w = 1*rand(1, mid_layer_sz+1);
nn_case2.func = @(x) tanh(x);
nn_case2.diff = @(x) 1 - tanh(x).^2;

nn_case2 = neuro_net_init(nn_case2);

train_par.max_it = 200;
train_par.max_error = 1e-5;

target.training = channel_samples_case1.desejado_treinamento;
target.validation =  channel_samples_case1.desejado_val;

[nn_t_case1, error, it_bfgs] = batch_cg_bfgs_training(channel_samples_case1.entrada_treinamento,  ...
                                                      target, ...
                                                      nn_case1,
                                                      train_par);

[nn_t_case2, error, it_bfgs] = batch_cg_bfgs_training(channel_samples_case2.entrada_treinamento,  ...
                                                      channel_samples_case2.desejado_treinamento, ...
                                                      nn_case2,
                                                      train_par);

map_fronteira_mlp_equalizacao(h_case1, d_case1, nn_t_case1.v', nn_t_case1.w, ...
                              channel_samples_case1.entrada_treinamento);

map_fronteira_mlp_equalizacao(h_case2, d_case2, nn_t_case2.v', nn_t_case2.w, ...
                              channel_samples_case2.entrada_treinamento);

% Generating error graph
case1_net_output = neural_nete(channel_samples_case1.entrada_teste(:, 1:1000), nn_case1);

% BER and MSE calculation
case1_net_output_bpsk = case1_net_output;
case1_net_output_bpsk(case1_net_output > 0) = 1; 
case1_net_output_bpsk(case1_net_output < 0) = -1; 

mse_12center = mean((channel_samples_case1.desejado_teste(:, 1:1000) - case1_net_output_bpsk).^2);
ber_12center = sum(case2_net_output_bpsk ~= channel_samples_case1.desejado_teste(:, 1:1000))/length(case1_net_output_bpsk);

% Generating error graph
case2_net_output = neural_nete(channel_samples_case2.entrada_teste(:, 1:1000), nn_case2);

% BER and MSE calculation
case2_net_output_bpsk = case2_net_output;
case2_net_output_bpsk(case2_net_output > 0) = 1; 
case2_net_output_bpsk(case2_net_output < 0) = -1; 

mse_2center = mean((channel_samples_case2.desejado_teste(:, 1:1000) - case2_net_output_bpsk).^2);
ber_2center = sum(case2_net_output_bpsk ~= channel_samples_case2.desejado_teste(:, 1:1000))/length(case2 _net_output_bpsk);

plot(error,'.-')
