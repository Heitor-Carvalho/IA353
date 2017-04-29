addpath('../LineSearchs/')
addpath('../BackPropagation/')
addpath('../NeuralNetwork/')
addpath('../TrainingMethods/')
%% Test 1 - XOR function

% Test one pattern at the time
train_set = [0 0; 1 1; 0 1; 1 0]';
target = [0 0 1 1];

% Neural network structure
clear nn
in_sz = 2;
mid_layer_sz = 5;
out_sz = 1;
nn.b = 1;
nn.v = 2*rand(in_sz+1, mid_layer_sz);
nn.w = 2*rand(1, mid_layer_sz+1);
nn.func = @(x) exp(x)./(1 + exp(x));
nn.diff = @(x) exp(x)./(1 + exp(x)).^2;
nn = neuro_net_init(nn);

train_par.max_it = 200;
train_par.max_error = 1e-5;

[nn_t, error, it_bfgs] = batch_cg_dfp_training(train_set, target, nn, train_par);
nn_out = neural_nete(train_set, nn_t) 
[nn_t, error, it_dfp] = batch_dfp_training(train_set, target, nn, train_par);
nn_out = neural_nete(train_set, nn_t) 
[nn_t, error, it_fr] = batch_fr_training(train_set, target, nn, train_par);
nn_out = neural_nete(train_set, nn_t) 
[nn_t, error, it_pr] = batch_pr_training(train_set, target, nn, train_par);
nn_out = neural_nete(train_set, nn_t) 
[nn_t, error, it_oss] = batch_oss_training(train_set, target, nn, train_par);
nn_out = neural_nete(train_set, nn_t) 
[nn_t, error, it_lm] = batch_lm_training(train_set, target, nn, train_par, 1e-5);
nn_out = neural_nete(train_set, nn_t) 
[nn_t, error, it_lm] = batch_gradient_lin_search_training(train_set, target, nn, train_par, 1e-3);
nn_out = neural_nete(train_set, nn_t) 

%% Test 2 - Polinomial interpolation

% Training pattern
in_ref = linspace(0, 5, 50);
target_ref = in_ref.^2 - 10*sin(in_ref).^2 + 3;
target_ref = target_ref/max(target_ref) - mean(target_ref);

down_sample_factor = 1;
in = downsample(in_ref, down_sample_factor);
target = downsample(target_ref, down_sample_factor);

% Neural network structure
clear nn
in_sz = 1;
mid_layer_sz = 20;
out_sz = 1;
nn.v = 1*rand(in_sz+1, mid_layer_sz);
nn.w = 1*rand(1, mid_layer_sz+1);
nn.b = 1;
nn.func = @(x) exp(x)./(1 + exp(x));
nn.diff = @(x) exp(x)./(1 + exp(x)).^2;
nn = neuro_net_init(nn);

train_par.max_error = 1e-4;
train_par.max_it = 200;

[nn_t, error, it] = batch_bfgs_training(in, target, nn, train_par);
[nn_t, error, it] = batch_lm_training(in, target, nn, train_par, 1e-2);

nn_out = neural_nete(in_ref, nn_t);  

figure(2)
plot(in_ref, target_ref, 'o')
hold on
plot(in_ref, nn_out, '.')
err = mean((nn_out - target_ref).^2)

%% Test 3 - sin(x).*cos(2*x)

train_set = linspace(0, 2*pi, 40);
target = sin(train_set).*cos(2*train_set);

% Neural network structure
clear nn
in_sz = 1;
mid_layer_sz = 10;
out_sz = 1;
nn.v = 1*rand(in_sz+1, mid_layer_sz);
nn.w = 1*rand(1, mid_layer_sz+1);
nn.b = 1;
nn.func = @(x) exp(x)./(1 + exp(x));
nn.diff = @(x) exp(x)./(1 + exp(x)).^2;
nn = neuro_net_init(nn);

train_par.max_error = 1e-4;
train_par.max_it = 200;

[nn_t, error, it] = batch_bfgs_training(train_set, target, nn, train_par);

plot(neural_nete(train_set, nn_t));
