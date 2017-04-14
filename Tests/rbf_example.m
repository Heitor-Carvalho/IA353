addpath('../NeuralNetwork/')
addpath('../RadialBasisNetwork/')
%% Test 1 - XOR function

% Training pattern
train_set = [0 0; 1 1; 0 1; 1 0]';
target = [0 0 1 1];

% Calculating RBF network
middle_layer_sz = 3;
regulatization = 0;
nn = calc_rbf_network(train_set, target, middle_layer_sz, regulatization);

% Testing RBF network
neural_nete_rbf(train_set, nn)



%% Test 2 - Sinc function interpolation

% Training pattern
train_set = linspace(-5, 5, 40);
target = sinc(train_set);

% Calculating RBF network
middle_layer_sz = 20;
regulatization = 0;
nn = calc_rbf_network(train_set, target, middle_layer_sz, regulatization);

% Testing RBF network - Calculation output
out = neural_nete_rbf(train_set, nn);

figure(1)
plot(train_set, target, 'o')
hold on
plot(train_set, out, '.')
err = mean((out - target).^2);

%% Test 3 - Polinomial interpolation

% Training pattern
in_ref = linspace(0, 5, 50);
target_ref = in_ref.^2 - 10*sin(in_ref).^2 + 3;

down_sample_factor = 1;
train_set = downsample(in_ref, down_sample_factor);
target = downsample(target_ref, down_sample_factor);

% Neural network structure
% Calculating RBF network
middle_layer_sz = 8;
regulatization = 0;
nn = calc_rbf_network(train_set, target, middle_layer_sz, regulatization);

% Testing RBF network - Calculation output
out = neural_nete_rbf(train_set, nn);

figure(1)
plot(train_set, target, 'o')
hold on
plot(train_set, out, '.')
err = mean((out - target).^2);
