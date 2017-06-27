function [H] = get_elm_net_regression_matrix(in, target, nn)
 % [H] = get_elm_net_regression_matrix(in, target, nn) - Return the 
 % the H matrix used in the regression problem to determine the 
 % ELM output  weigths.
 % Inputs:
 %   in       : input samples (One or multiples samples)
 %   target   : input samples (One or multiples samples)
 % Initialized neuro network structure:
 %   nn.v     : middle layer weights
 %   nn.w     : output layer weights
 %   nn.b     : neurons bias
 %   nn.func  : neural network function 
 %
 % Outputs:
 %   H        : output layer weigths Matrix
 
  [in_sz, samples_sz] = size(in);
  middle_sz = size(nn.v, 2);
  
  [~, mid_layer_out_bias] = neural_nete(in, nn);

  H = reshape(mid_layer_out_bias, middle_sz+1, samples_sz)';
  
end
