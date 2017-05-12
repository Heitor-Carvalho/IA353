function [w] = calc_esn_weigths(in, target, reg_fact, nn_feedback)
% This training procedure considers feedback from only the network states
% and not the network outputs.
  
  % Calculating the target sample size
  samples_sz = length(target);
  
  % Starting echo state network with state 0
  states = zeros(1, nn_feedback.mid_sz);
  
  % States holder 
  states_holder = zeros(nn_feedback.mid_sz, length(target));

  % Generate states sequencies
  for i=1:length(target)
    [~, ~, states] = neural_nete([in(:, i); transpose(states)], nn_feedback);
    states_holder(:, i) = transpose(states);
  end

  H = [nn_feedback.b*ones(1, length(target)); states_holder]';
  if(nn_feedback.mid_sz+1 > samples_sz)
      w = H'*pinv(H*H' + reg_fact*eye(min(size(H))))*target';
  else
      w = pinv(H'*H + reg_fact*eye(min(size(H))))*H'*target';
  end
  w = transpose(w);

end
