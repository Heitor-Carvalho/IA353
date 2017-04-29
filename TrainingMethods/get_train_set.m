function [training_set, input_sets] = get_train_set(input_sets)
% Return only the training set from the input_sets
%
% Inputs:
%
% input_sets : variable containing differentes input sets, 
% like training, validation and possible other sets. Each one
% corresponding to a element the input_sets strucuture.
%
% OBS: The first element is consedered the training set

if(iscell(input_sets))
  training_set = input_sets{1};
else
  training_set = input_sets;
  input_sets = mat2cell(input_sets, size(input_sets, 1));
end

end
