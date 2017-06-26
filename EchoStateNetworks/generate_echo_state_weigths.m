function [Win, Wfb, Wall] = generate_echo_state_weigths(input_par, feedback_par)
% generate_echo_state_weights: Function to generate the input 
% and feedback weigth matrix for the echo state network
%
% Inputs:
%   input_par: Parameters to generate the echo state network input 
%              weights.
%     input_par.sparseness: Percentage of connections different 
%                           than zero. Number from 0 to 1
%     input_par.range: Range o values used to generate the weights
%                      values.
%     input_par.sz: Size of the input weight matrix
%
%   feedback_par: Parameters to generate the echi state network feedback
%                 weigths.
%     feedback_par.sparseness: Percentage of connections different than 
%                              zero 
%     feedback_par.range: Range of values used to generate the weights 
%                         values
%     feedback_par.sz: Size of the feedback weigth matrix
%     feedback_par.alpha (optional): Parameter to adjust the network 
%                        dynamic responde. High alpha for lower speeds
%                        dynamics. Low alpha for higher speed dynamics
%
% Outputs:
%   Win: Input weigth matrix
%   Wfb: Feedback weigth matrix
%   Wall: Combined Win and Wfb matrix

  if(isfield(input_par, 'sparseness'))
    Win = full(sprand(input_par.sz(1)+1, input_par.sz(2), input_par.sparseness));
    Win(Win ~= 0) = 2*(Win(Win ~= 0) - 1)*input_par.range;
  else
    Win = 2*rand(input_par.sz(1)+1, input_par.sz(2))-1;
  end
  
  if(isfield(feedback_par, 'sparseness'))
    Wfb = full(sprand(feedback_par.sz(1), feedback_par.sz(2), feedback_par.sparseness));
    Wfb(Wfb ~= 0) = 2*(Wfb(Wfb ~= 0) - 0.5)*feedback_par.range;
  else
    Wfb = 2*rand(feedback_par.sz(1), feedback_par.sz(2))-1;
  end
  
  % Adjusting weigths spectral radius - An heuristic procedure
  % to adjust the echo state net dynamics
  if(isfield(feedback_par, 'alpha'))
    lambdas = eig(Wfb);
    Wfb = Wfb/max(abs(lambdas));
    Wfb = Wfb*feedback_par.alpha;
  end
  Wall = [Win; Wfb];

end
