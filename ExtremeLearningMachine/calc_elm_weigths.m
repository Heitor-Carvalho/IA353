function [w] = calc_elm_weigths(in, target, reg_fact, nn)
 % calc_elm_weigths(in, target, c, nn) - Calculates the output layer weigths
 % that minimizes the MSE with regularization parameter c
 % for a given intermediate layer and a training set.
 % Inputs:
 %   in       : input samples (One or multiples samples)
 %   target   : input samples (One or multiples samples)
 %   reg_fact : regularization parameters
 % Initialized neuro network structure:
 %   nn.v     : middle layer weights
 %   nn.w     : output layer weights
 %   nn.b     : neurons bias
 %   nn.func  : neural network function 
 %
 % Outputs:
 %   w        : output layer weigths
 % To do: Add support to multiple output layers
 
 
  [in_sz, samples_sz] = size(in);
  middle_sz = size(nn.v, 2);
  
  [~, mid_layer_out_bias] = neural_nete(in, nn);

  H = reshape(mid_layer_out_bias, middle_sz+1, samples_sz)';
  if(middle_sz+1 > samples_sz)
      w = H'*pinv(H*H' + reg_fact*eye(min(size(H))))*target';
  else
      w = pinv(H'*H + reg_fact*eye(min(size(H))))*H'*target';
  end
  
end