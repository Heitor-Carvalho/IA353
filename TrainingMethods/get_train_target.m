function [training_target, target_sets] = get_train_target(target_sets)
% Return only the training set from the tartget_sets
% 
% Inputs:
%
% target_sets : variable containing differet target sets,
% like training, validation and possible other sets. Each one
% corresponding to a element in the output_sets structure.
%
% OBS: The first element is considered the training set

if(iscell(target_sets))
  training_target = target_sets{1};
else
  training_target = target_sets;
  target_sets = mat2cell(target_sets, size(target_sets,1));
end

end
