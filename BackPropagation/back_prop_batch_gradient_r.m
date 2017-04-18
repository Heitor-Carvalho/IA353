function [grad] = back_prop_batch_gradient(train_set, target, nn)
  % back_prop_batch_gradient(train_set, target, nn) - Calculate the neural 
  % network gradient and Jacobian using the backpropagation algorithm. In 
  % this function only a one hiddnel layer neural network is considered.
  % Inputs:
  % train_set : Patterns used to train the neuro network
  % target    : Reference values to the patterns
  % Initialized neuro network structure:
  %   nn.v    : middle layer weights
  %   nn.w    : output layer weights
  %   nn.b    : neurons bias
  %   nn.func : neuron activation function 
  %   nn.diff : neuron activation function derivate
  %
  % Outputs:
  % grad      : Neural network weights gradient
  % J         : Neural network weights Jacobian
  %
  % To do : add support to multiple outputs

  samples_sz = size(train_set, 2);
  
  % Foward part - Neural network output
  [R_y_k, y_k, R_z_jb, z_jb, R_z_j, R_a_j, a_j, x_in] = neural_nete_r(train_set, nn, vnet);  

  % Neuro network error
  delta_k = target - y_k;
  delta_k = reshape(delta_k, 1 , 1, samples_sz);
  R_delta_k = reshape(R_y_k, 1 , 1, samples_sz);
      
  % Output layer weights (Linear combiner)
  dEdW = 2*repmat(delta_k, 1, nn.mid_sz+1).*z_jb;
  R_dEdW = R_delta_k.*z_jb + delta_k.*R_z_jb;
  
  dEdV = mean(dEdV, 3);
  R_dEdV = mean(R_dEdV, 3);
    
  % Middle layer weights
  w = repmat(nn.w(:, 2:end), 1, 1, samples_sz);
    
  delta_j = -repmat(nn.diff(a_j), nn.in_sz+1, 1)             .* ...
             repmat(w, nn.in_sz+1, 1)                        .* ...
             repmat(delta_k, nn.in_sz+1, nn.mid_sz);

  R_delta_j = -repmat(nn.diff(a_j), nn.in_sz+1, 1)           .* ...
               repmat(vnet.v, nn.in_sz+1, 1)                 .* ...
               repmat(delta_k, nn.in_sz+1, nn.mid_sz);
              -repmat(nn.diff(a_j), nn.in_sz+1, 1)           .* ...
               repmat(w, nn.in_sz+1, 1)                      .* ...
               repmat(R_delta_k, nn.in_sz+1, nn.mid_sz);
              -repmat(nn.ddiff(a_j), nn.in_sz+1, 1)          .* ...
               repmat(w, nn.in_sz+1, 1)                      .* ...
               repmat(delta_k, nn.in_sz+1, nn.mid_sz)        .* ...
               repmat(R_aj, nn.in_sz+1, nn.mid_sz);

  dEdW = 2*delta_j.*repmat(in_bias, 1, nn.mid_sz);
  dEdW = mean(dEdW, 3);
  
  grad_v = dEdV(:);
  grad_w = dEdW(:);

  grad = [grad_v; grad_w];
  
end
