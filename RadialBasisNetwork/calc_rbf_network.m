function [nn] = calc_rbf_network(train_set, target, middle_layer_sz, reg_fact)
 % calc_rbf_network(train_set, target, middle_layer_sz, reg_fact)
 % Calculates the output layer weigths of a RBF network that minimizes 
 % the MSE with regularization parameter c for a given intermediate 
 % layer and a training set.
 % Inputs:
 %   train_set         : input samples (One or multiples samples)
 %   target            : input samples (One or multiples samples)
 %   middle_layer_sz   : input samples (One or multiples samples)
 %   reg_fact          : regularization parameters
 %
 % Outputs:
 %   nn                : RBF neuro network
 % To do: Add support to multiple output layers
 
  [in_sz, samples_sz] = size(train_set);
  
  % Gaussian function
  nn.func = @(x) exp(-x.^2);
  
  % Calculating gaussian centroids and variance
  [idxs, centroids] = kmeans(train_set', middle_layer_sz);
  nn.b = 0;
  nn.c = centroids';
  nn.sig = sqrt(max(pdist(nn.c'))); 

  % Preparing neuro network
  nn.v = [zeros(1, middle_layer_sz); ones(in_sz, middle_layer_sz)];
  nn.b = 0;
  nn = neuro_net_init(nn);

  [~, mid_layer_out_bias] = neural_nete_rbf(train_set, nn);

  H = reshape(mid_layer_out_bias, middle_layer_sz+1, samples_sz)';
  if(middle_layer_sz > samples_sz)
    nn.w = transpose(H'*pinv(H*H' + reg_fact*eye(min(size(H))))*target');
  else
    nn.w = transpose(pinv(H'*H + reg_fact*eye(min(size(H))))*H'*target');
  end
  
end