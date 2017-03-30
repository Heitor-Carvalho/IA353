%% Test 1 - XOR function

% Test one pattern at the time
train_set = [0 0; 1 1; 0 1; 1 0]';
target = [0 0 1 1];

% Neural network structure
clear nn
in_sz = 2;
mid_layer_sz = 4;
out_sz = 1;
nn.func = @(x) 1 ./ (1 + exp(-x));
nn.b = 1;
nn.v = 1*randn(in_sz+1, mid_layer_sz);
nn.w = 1*randn(1, mid_layer_sz+1);
nn.func = @logsig;
nn.diff = @(x) exp(x)./(1 + exp(x)).^2;
nn = neuro_net_init(nn);

train_par.alpha = 0.01;
train_par.max_error = 1e-5;
train_par.max_it = 1e3;
train_par.beta = 0.01;

[nn_t, error, it] = back_prop(train_set, target, nn, train_par);

nn_out = neural_nete(train_set, nn_t);  

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
nn.v = 1*randn(in_sz+1, mid_layer_sz);
nn.w = 1*randn(1, mid_layer_sz+1);
nn.b = 1;
nn.func = @(x) 1./(1+exp(-x));
nn.diff = @(x) exp(x)./(1 + exp(x)).^2;
nn = neuro_net_init(nn);

train_par.alpha = 0.2;
train_par.max_error = 1e-4;
train_par.max_it = 2e3;
train_par.beta = 0;
keyboard
[nn_t, error, it] = back_prop(in, target, nn, train_par);

nn_out = neural_nete(in_ref, nn_t);  

figure(2)
plot(in_ref, target_ref, 'o')
hold on
plot(in_ref, nn_out, '.')
err = mean((nn_out - target_ref).^2)

