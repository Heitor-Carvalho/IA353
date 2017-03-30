function [out, mid_layer_func_out_bias, mid_layer_func_out, mid_layer_func_in, in_bias] = neural_nete(in, nn)
  % neural_net(in, v, w, b, func) - Calculate a one hidden layer neural network
  %   in   : input samples (One or multiples samples)
  % Initialized neuro network structure:
  %   nn.v    : middle layer weights
  %   nn.w    : output layer weights
  %   nn.b    : neurons bias
  %   nn.func : neural network function 
  %
  %  OBS:  The samples must passed as collumns where each collumns corresponds to
  % a sample and each line correspond to one different input.
  %        The middle layer weigths for each input (include the bias input)
  %        correspond to one line of the matrix v. 
  %        The sampe happens with the output weigths
  
  
  % Adding bias to the neuro network input
  in_bias = [nn.b*ones(1, size(in, 2)); in];
  [in_sz, samples_sz] = size(in_bias);
  in_bias = reshape(in_bias, in_sz, 1, samples_sz);
  
  middle_sz = size(nn.v, 2);
  
  % Checking for neuro network weitghs 
  if(~isfield(nn, 'w'))
    nn.w = zeros(1, middle_sz+1);
  end
  out_sz = size(nn.w, 1);
  
  % Checking middle layer neurons number
  assert(in_sz == nn.in_sz+1, 'Unexpected number of neurons (collumns) for v, should be %d', in_sz);
  
  mid_layer_sum_in = repmat(in_bias, 1, middle_sz).*repmat(nn.v, 1, 1, samples_sz);
  mid_layer_func_in = sum(mid_layer_sum_in, 1);
  mid_layer_func_out = nn.func(mid_layer_func_in);
  
  % Adding output layer bias
  mid_layer_func_out_bias = [nn.b*ones(1, 1, size(in, 2)) mid_layer_func_out];
  
  % Calculating outputs
  out = sum(repmat(mid_layer_func_out_bias, out_sz, 1).*repmat(nn.w, 1, 1, samples_sz), 2);
  out = reshape(out, out_sz, samples_sz);
  
end