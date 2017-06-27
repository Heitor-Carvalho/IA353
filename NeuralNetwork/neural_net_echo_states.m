function [out] = neural_net_echo_states(in, nn)
% [out] = neural_net_echo_states(in, nn) - Calculates the output
% for the a echo state  neural network.
%
% Inputs:
%  in - neural network training set
%  nn - echo state network structure, the weithgs in the middle
% layers must follow the sequence:
%    - nn.v = [bias weights, input_weights, echo_states_weights]
%
% Outpus:
%  out - neural network output

  % Inital state set to zero
  states = zeros(1, nn.mid_sz);

  % Output placeholder
  out = zeros(1, length(in));
  
  for i = 1:length(in)
    [out(i), ~, states] = neural_nete([in(:, i); transpose(states)], nn);
  end

end
