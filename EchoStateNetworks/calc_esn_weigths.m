function calc_esn_weigths(in, target, reg_fact, nn_feedback)
% This training procedure considers feed from only the network states
% and not the network outputs.

  % Starting echo state network with state 0
  states = zeros();

  % Generate states sequencies
  for i=1:length(target)
    feedback_nn_input = [in(:, 1) states]  
    [~, state] = neuro_net(feedback_input, nn)
  end

  % Given the inputs and the states, calculate the output weitghs
  if()
  //pseudoinverse
  end

end
