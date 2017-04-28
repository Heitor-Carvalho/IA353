function [training_set] = get_train_set(input_sets)
% Return only the training set from the train_sets 
%
% Inputs:
%
% input_sets : variable containing differentes input sets, 
% like training, validation and possible other sets. Each one
% corresponding to a page in the input_set variable.
%
% OBS: The first page is consedered the training set

training_set = input_sets(:, :, 1);


end
