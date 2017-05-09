% Network - Input layer size
net_in_sz = 2;

% Network - Middle layer size 
net_middle_sz = 10;

% Network - Output layer size
net_out_sz = 1;

% Creating the weigths to the 
% the non feedback echo state network
input_par.sz = [in_sz middle_sz];
input_par.range = 1;
input_par.sparseness = 1;

feedback_par.sz = [middle_sz middle_sz];
feedback_par.range = 1;
feedback_par.alpha = 0.95;

[~, ~, Weigths] = generate_echo_state_weigths(input_par, feedback_par);

nn.v = Weigths;
nn.b = 0;
nn = neural_net_init(nn);

// Call training routine !


