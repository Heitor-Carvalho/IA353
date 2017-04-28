function [training_target] = get_train_target(target_sets)
% Return only the training set from the tartget_sets
% 
% Inputs:
%
% target_sets : variable containing differet target sets,
% like training, validation and possible other sets. Each one
% corresponding to a page in the output_sets variable.
%
% OBS: The first page is considered the training set

training_target = target_sets(:, :, 1);

end
