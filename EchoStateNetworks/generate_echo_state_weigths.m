function [Win, Wfb, Wall] = generate_echo_state_weigths(input_par, feedback_par)
% generate_echo_state_weigths : Function to generate the input 
% and feedback weigth matrix for the echo state network
%
% Inputs:
%   input_par: Parameters to generate the echo state network input 
%              weigths.
%     input_par.sparseness: Percentage of connections different 
%                           than zero. Number from 0 to 1
%     input_par.range: Range o values used to generate the weigths
%                      values.
%     input_par.sz: Size of the input weigth matrix
%
%   feedback_par: Parameters to generate the echi state network feedback
%                 weigths.
%     feedback_par.sparseness: Percentage of connections different than 
%                              zero 
%     feedback_par.range: Range of values used to generate the weigths 
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

  Win = full(sprand(input_par.sz(1), input_par.sz(2), input_par.sparseness));
  Win(Win ~= 0) = 2*(Win(Win ~= 0) - 0.5)*input_par.range;

  Wfb = full(sprand(feedback_par.sz(1), feedback_par.sz(2), feedback_par.sparseness));
  Wfb(Wfb ~= 0) = 2*(Wfb(Wfb ~= 0) - 0.5)*feedback_par.range;
 
  % Adjusting weigths spectral radius - An heuristic procedure 
  % to adjust the echo state net dynamics
  if(isfield(feedback_par, 'alpha'))
    lambdas = eig(Wfb);
    Wfb = Wfb/max(abs(lambdas));
    Wfb = Wfb*feedback_par.alpha;
  end
 
  Wall = [Win; Wfb];

end
