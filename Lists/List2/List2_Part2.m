addpath('../NeuralNetwork/')

% Loading training data
load('./ListData/channel_training_data.mat')

% Loading channels

% Case 1
h_case1 = [0.9 1];                          % Channel
d_case1 = 0;                                % Delay

% Case 2
h_case2 = [1 0.5 0.2];                      % Channel
d_case2 = 2;                                % Delay

% Generating random neural network - Case 1
clear nn_case1
in_sz = 2;
mid_layer_sz = 10;
out_sz = 1;
nn_case1.b = 1;
nn_case1.v = 1*randn(in_sz+1, mid_layer_sz);
nn_case1.w = 1*randn(1, mid_layer_sz+1);
nn_case1.func = @(x) tanh(x);
nn_case1.diff = @(x) 1 - tanh(x).^2;

nn_case1 = neuro_net_init(nn_case1);

[w1,w2] = meshgrid(-10:0.5:10);
w1_vector = w1(:);
w2_vector = w2(:);

J = zeros(size(w1_vector));

neuron_one = randi([1 8], 1, 1); 
neuron_two = randi([1 8], 1, 1);
input_one = randi([1 2], 1, 1);
input_two = randi([1 3], 1, 1);

% Generating error graph
for i = 1:length(w1_vector)
  nn_case1.v(input_one, neuron_one) = w1_vector(i);
  nn_case1.v(input_two, neuron_two) = w2_vector(i);
  case1_net_output = neural_nete(channel_samples_case1.entrada_teste(:, 1:1000), nn_case1);
  J(i) = mean((channel_samples_case1.desejado_teste(:, 1:1000) - case1_net_output).^2);
end

J = reshape(J, size(w1));

mesh(w1, w2, J)