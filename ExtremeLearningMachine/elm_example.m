%% Test 1 - XOR function

% Training pattern
in = [0 0; 1 1; 0 1; 1 0]';
target = [0 0 1 1];

% Neural network structure
clear nn
in_sz = 2;
mid_layer_sz = 12;
out_sz = 1;
nn.func = @(x) 1 ./ (1 + exp(-x));
nn.b = 1;
nn.v = 1*randn(in_sz+1, mid_layer_sz);
nn = neuro_net_init(nn);

err = [];
for c = 0:0.1:0
  nn.w = elm_weigths(in, target, c, nn)';
  err = [err, mean((neural_nete(in, nn) - target).^2)];
end


%% Test 2 - Sinc function interpolation

% Training pattern
in = linspace(-5, 5, 40);
target = sinc(in);

% Neural network structure
clear nn
nn.v = 1*randn(2, 200);
nn.func = @(x) 1./(1+exp(-x));
nn = neuro_net_init(nn);
nn.b = 1;

% Call training routine
w = elm_weigths(in, target, 0, nn, 1);

% Setting up weigths
nn.w = w';

% Calculation output
out = neural_nete(in, nn);

figure(1)
plot(in, target, 'o')
hold on
plot(in, out, '.')
err = mean((out - target).^2);

%% Test 3 - Polinomial interpolation

% Training pattern
in_ref = linspace(0, 5, 50);
target_ref = in_ref.^2 - 10*sin(in_ref).^2 + 3;

down_sample_factor = 1;
in = downsample(in_ref, down_sample_factor);
target = downsample(target_ref, down_sample_factor);

% Neural network structure
clear nn
nn.v = 1*randn(2, 10);
nn.b = 1;
nn.func = @(x) 1./(1+exp(-x));
nn = neuro_net_init(nn);

% Call training routine
c = 0.6e-5;
c = 0;
w = elm_weigths(in, target, c, nn);

% Setting up weigths
nn.w = w';

% Calculation output
out = neural_nete(in_ref, nn);

figure(2)
plot(in_ref, target_ref, 'o')
hold on
plot(in_ref, out, '.')
err = mean((out - target_ref).^2)
