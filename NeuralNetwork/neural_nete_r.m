function [R_y_k, R_z_jb, z_jb, R_z_j, R_a_j, x_in] = neural_nete_r(in, nn, vnet)
  % neural_net(in, v, w, b, func) - Calculate a one hidden layer neural network
  % Inputs:
  %   in      : input samples (One or multiples samples)
  % Initialized neuro network structure:
  %   nn.v    : middle layer weights
  %   nn.w    : output layer weights
  %   nn.b    : neurons bias
  %   nn.func : neuron activation function
  %
  % Outputs:
  %   out                     : neural network output
  %   mid_layer_func_out_bias : middle layer output after activation function plus bias
  %   mid_layer_func_out      : middle layer output after activation function
  %   mid_layer_func_in       : middle layer activation function input
  %   in_bias                 : input plus bias
  %
  %  OBS:  The samples must passed as collumns where each collumns corresponds to
  %        a sample and each line correspond to one different input.
  %        The middle layer weigths for each input (include the bias input)
  %        correspond to one line of the matrix v. 
  %        The sampe happens with the output weigths
  
  
  % Adding bias to the neuro network input
  x_in = [nn.b*ones(1, size(in, 2)); in];
  [in_sz, samples_sz] = size(x_in);
  x_in = reshape(x_in, in_sz, 1, samples_sz);
  
  middle_sz = size(nn.v, 2);
  
  % Checking for neuro network weitghs 
  if(~isfield(nn, 'w'))
    nn.w = zeros(1, middle_sz+1);
  end
  out_sz = size(nn.w, 1);
  
  % Checking middle layer neurons number
  assert(in_sz == nn.in_sz+1, 'Unexpected number of neurons (collumns) for v, should be %d', in_sz);
  
  a_jm = repmat(x_in, 1, middle_sz).*repmat(nn.v, 1, 1, samples_sz);
  a_j = sum(a_jm, 1);
  
  R_a_jm = repmat(x_in, 1, middle_sz).*repmat(vnet.v, 1, 1, samples_sz);
  R_a_j = sum(R_a_jm, 1);

  z_j = nn.func(a_j);
  R_z_j = nn.diff(a_j).*R_a_j;
  
  % Adding output layer bias
  z_jb = [nn.b*ones(1, 1, size(in, 2)) z_j];
  R_z_jb = [nn.b*ones(1, 1, size(in, 2)) R_z_j];
  
  % Calculating outputs
  y_k = sum(repmat(z_jb, out_sz, 1).*repmat(nn.w, 1, 1, samples_sz), 2);

  R_y_k = sum(repmat(R_z_jb, out_sz, 1).*repmat(nn.w, 1, 1, samples_sz) +...
            repmat(z_jb, out_sz, 1).*repmat(vnet.w, 1, 1, samples_sz), 2);
    
  y_k = reshape(y_k, out_sz, samples_sz);
  R_y_k = reshape(R_y_k, out_sz, samples_sz);
  
end