function [mse_errors] = get_mse_error(input_sets, target_sets, nn)
% Calculate the MSE erro for all the sets in the input_sets using
% its corresponding sets in the target_sets
%
% Inputs:
%
% training_set : input_sets used in MSE calculation
% target_sets  : target sets for the input_sets used
% nn           : neurak network struture.

  mse_errors = zeros(size(target_sets, 2), 1);
  for i = 1:size(target_sets, 2)
    mse_errors(i) = mean((target_sets{i} - neural_nete(input_sets{i}, nn)).^2, 2);
  end

end
