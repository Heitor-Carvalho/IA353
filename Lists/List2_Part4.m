addpath('../RadialBasisNetwork/')
addpath('../NeuralNetwork/')
addpath('./ThirdParty/')

% Loading training data
load('channel_training_data.mat')

% RBF training - Case 1 - Exact number of centrers 
nn_case1_1 = calc_rbf_network(channel_samples_case1.entrada_treinamento, channel_samples_case1.desejado_treinamento, 8, 0);

% RBF training - Case 2 - 4 extra centers
nn_case1_2 = calc_rbf_network(channel_samples_case1.entrada_treinamento, channel_samples_case1.desejado_treinamento, 12, 0);

% Case 1
h_case1 = [0.9 1]                           % Channel
d_case1 = 0;                                % Delay

% Case 1
h_case2 = [1 0.5 0.2]                       % Channel
d_case2 = 2;                                % Delay

% Ploting RBF founded centers:
figure(1)
plot(nn_case1_1.c(1,:), nn_case1_1.c(2,:),'o')
grid
figure(2)
plot(nn_case1_2.c(1,:), nn_case1_2.c(2,:),'o')
grid

% Generating map using ThirdParty code
map_fronteira_mlp_equalizacao_rbf(h_case1, d_case1, nn_case1_1.v, nn_case1_1.w, nn_case1_1.c, nn_case1_1.sig, channel_samples_case1.entrada_treinamento);
map_fronteira_mlp_equalizacao_rbf(h_case1, d_case1, nn_case1_2.v, nn_case1_2.w, nn_case1_2.c, nn_case1_2.sig, channel_samples_case1.entrada_treinamento);



