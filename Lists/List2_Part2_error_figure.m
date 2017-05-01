addpath('../NeuralNetwork/')

% Loading training data
load('./ListData/channel_training_data.mat')

% Loading network
load('./ListData/Part2/error_network_part2')

[w1,w2] = meshgrid(-10:0.5:10);
w1_vector = w1(:);
w2_vector = w2(:);

Jmax = zeros(size(w1_vector));
Jmin = zeros(size(w1_vector));

% Generating error graph
for i = 1:length(w1_vector)
  nn_case1_max.v(2, 2) = w1_vector(i);
  nn_case1_max.v(2, 5) = w2_vector(i);
  nn_case1_min.v(2, 1) = w1_vector(i);
  nn_case1_min.v(1, 5) = w2_vector(i);
  case1_net_output_max = neural_nete(channel_samples_case1.entrada_teste(:, 1:1000), nn_case1_max);
  Jmax(i) = mean((channel_samples_case1.desejado_teste(:, 1:1000) - case1_net_output_max).^2);
  case1_net_output_min = neural_nete(channel_samples_case1.entrada_teste(:, 1:1000), nn_case1_min);
  Jmin(i) = mean((channel_samples_case1.desejado_teste(:, 1:1000) - case1_net_output_min).^2);
end

Jmax = reshape(Jmax, size(w1));
Jmin = reshape(Jmin, size(w1));

figure(1)
mesh(w1, w2, Jmax)

figure(2)
mesh(w1, w2, Jmin)
