addpath('../RadialBasisNetwork/')
addpath('../NeuralNetwork/')
addpath('./ThirdParty/')

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

% RBF training - Case 1_1 - Exact number of centrers 
nn_case1_1 = calc_rbf_network(channel_samples_case1.entrada_treinamento, channel_samples_case1.desejado_treinamento, 8, 0);

% RBF training - Case 1_2 - 4 extra centers
nn_case1_2 = calc_rbf_network(channel_samples_case1.entrada_treinamento, channel_samples_case1.desejado_treinamento, 12, 0);

% RBF training - Case 2 - 16 centers
nn_case2 = calc_rbf_network(channel_samples_case2.entrada_treinamento, channel_samples_case2.desejado_treinamento, 16, 0);

map_fronteira_mlp_equalizacao_rbf(h_case1, d_case1, nn_case1_1.v, nn_case1_1.w, nn_case1_1.c, nn_case1_1.sig, channel_samples_case1.entrada_treinamento);
map_fronteira_mlp_equalizacao_rbf(h_case1, d_case1, nn_case1_2.v, nn_case1_2.w, nn_case1_2.c, nn_case1_2.sig, channel_samples_case1.entrada_treinamento);
map_fronteira_mlp_equalizacao_rbf(h_case2, d_case2, nn_case2.v, nn_case2.w, nn_case2.c, nn_case2.sig, channel_samples_case2.entrada_treinamento);

%% Loading ideal 8 center network

% Loading networks
nn_struct = load('neural_net_8centers_ideal')
nn_ideal = nn_struct.nn_case1_1;

% Generating mapping region
map_fronteira_mlp_equalizacao_rbf(h_case1, d_case1, nn_ideal.v, nn_ideal.w, nn_ideal.c, nn_ideal.sig, channel_samples_case1.entrada_treinamento);

% Generating error graph
ideal_8center_net_output = neural_nete_rbf(channel_samples_case1.entrada_teste(:, 1:1000), nn_ideal);

% BER and MSE calculation
ideal_8center_net_output_bpsk = ideal_8center_net_output;
ideal_8center_net_output_bpsk(ideal_8center_net_output > 0) = 1; 
ideal_8center_net_output_bpsk(ideal_8center_net_output < 0) = -1; 

mse_ideal_8center = mean((channel_samples_case1.desejado_teste(:, 1:1000) - ideal_8center_net_output).^2);
ber_ideal_8center = sum(ideal_8center_net_output_bpsk ~= channel_samples_case1.desejado_teste(:, 1:1000))/length(ideal_8center_net_output_bpsk);

%% Loading problem 8 centers network

% Loading networks
nn_struct = load('neural_net_8centers_problem')
nn_problem = nn_struct.nn_case1_1;

% Generating mapping region
map_fronteira_mlp_equalizacao_rbf(h_case1, d_case1, nn_problem.v, nn_problem.w, nn_problem.c, nn_problem.sig, channel_samples_case1.entrada_treinamento);

% Generating error graph
problem_8center_net_output = neural_nete_rbf(channel_samples_case1.entrada_teste(:, 1:1000), nn_problem);

% BER and MSE calculation
problem_8center_net_output_bpsk = problem_8center_net_output;
problem_8center_net_output_bpsk(problem_8center_net_output > 0) = 1; 
problem_8center_net_output_bpsk(problem_8center_net_output < 0) = -1; 

mse_problem_8center = mean((channel_samples_case1.desejado_teste(:, 1:1000) - problem_8center_net_output).^2);
ber_problem_8center = sum(problem_8center_net_output_bpsk ~= channel_samples_case1.desejado_teste(:, 1:1000))/length(problem_8center_net_output);

%% Loading problem 12 centers network

% Loading networks
nn_struct = load('neural_net_12centers')
nn_12_center = nn_struct.nn_case1_2;

% Generating mapping region
map_fronteira_mlp_equalizacao_rbf(h_case1, d_case1, nn_12_center.v, nn_12_center.w, nn_12_center.c, nn_12_center.sig, channel_samples_case1.entrada_treinamento);

% Generating error graph
center12_net_output = neural_nete_rbf(channel_samples_case1.entrada_teste(:, 1:1000), nn_12_center);

% BER and MSE calculation
center12_net_output_bpsk = center12_net_output;
center12_net_output_bpsk(center12_net_output > 0) = 1; 
center12_net_output_bpsk(center12_net_output < 0) = -1; 

mse_12center = mean((channel_samples_case1.desejado_teste(:, 1:1000) - center12_net_output).^2);
ber_12center = sum(center12_net_output_bpsk ~= channel_samples_case1.desejado_teste(:, 1:1000))/length(center12_net_output_bpsk);

%% Loading case 2 network

% Generating mapping region
map_fronteira_mlp_equalizacao_rbf(h_case2, d_case2, nn_case2.v, nn_case2.w, nn_case2.c, nn_case2.sig, channel_samples_case2.entrada_treinamento);

% Generating error graph
case2_net_output = neural_nete_rbf(channel_samples_case2.entrada_teste(:, 1:1000), nn_case2);

% BER and MSE calculation
case2_net_output_bpsk = case2_net_output;
case2_net_output_bpsk(case2_net_output > 0) = 1; 
case2_net_output_bpsk(case2_net_output < 0) = -1; 

mse_12center = mean((channel_samples_case2.desejado_teste(:, 1:1000) - case2_net_output_bpsk).^2);
ber_12center = sum(case2_net_output_bpsk ~= channel_samples_case2.desejado_teste(:, 1:1000))/length(case2_net_output_bpsk);

