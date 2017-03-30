function [w] = elm_weigths(in, target, c, nn, bias_use)
 % elm_weigths - Calculates the output layer weigths
 % that minimizes the MSE with regularization parameter c
 % for a given intermideate layer and a training set.
 
 % Falta multiplas saidas
  
  [in_sz, samples_sz] = size(in);
  middle_sz = size(nn.v, 2);
  
  [~, mid_layer_out_bias] = neural_nete(in, nn);

  H = reshape(mid_layer_out_bias, middle_sz+1, samples_sz)';
  
  if(middle_sz > samples_sz)
      w = H'*pinv(H*H' + c*eye(min(size(H))))*target';
  else
      w = pinv(H'*H + c*eye(min(size(H))))*H'*target';
  end
  
end