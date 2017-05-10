function [out] = neural_net_echo_states(in, nn)
  
  % Inital state set to zero
  states = zeros(1, nn.mid_sz);

  % Output placeholder
  out = zeros(1, length(in));
  
  for i = 1:length(in)
    [out(i), ~, states] = neural_nete([in(:, i); transpose(states)], nn);
  end

end