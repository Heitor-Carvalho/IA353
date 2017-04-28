function [nn_, err_hist, it] = batch_fr_training(train_set, target, nn, train_par)
  % To do: add support to multiples outputs
  % To do: add function description

  nn_ = nn;
  mse_error = train_par.max_error;
  err_hist = zeros(1, train_par.max_it);
  it = 0;

  % Calculate gradient using backpropagation
  g_i = -back_prop_batch_gradient(train_set, target, nn_);
  d = g_i; 
  samples_sz = size(train_set, 2);
  
  sig_ = 1e-8;
  lambda = 1e-5;
  lambda_ = 0;
  success = 1;
  keyboard
  while(mse_error >= train_par.max_error && ...
        it        < train_par.max_it)
   
    mse_error = 0;
  
        
%     % Functional to be minimized
%     Jfunc = @(alpha) mean((target - neural_nete(train_set, convert_w_to_neuronet_vw(weigths + alpha*d, nn_))).^2);

%     % Line search for alpha
%     alpha = golden_search(0, 10, Jfunc, 1e-3);

%     % Training method Fletcher-Reeves
    
      % Get weiths from neuro network structure    
    
      if(success)
        % Calculation second order information
        sig = sig_/sqrt(d'*d);
        weigths = convert_neuronet_vw_to_w(nn_);
        weigths_sig = weigths + sig*d;
        nn_sig = convert_w_to_neuronet_vw(weigths_sig, nn_);
        g_i1 = -back_prop_batch_gradient(train_set, target, nn_sig);
        s = (g_i1-g_i)/sig;
        delta = d'*s;
      end
      
%       delta = delta + (lambda - lambda_)*(d'*d);
      if(delta <= 0)
        lambda_ = 2*(lambda - delta/(d'*d));
        delta = -delta + lambda*(d'*d);
        lambda = lambda_;
      end
      % Calculation the step alpha
      u = d'*g_i;
      alpha = u/delta;

      % Functional to be minimized
      Jfunc = @(alpha) mean((target - neural_nete(train_set, convert_w_to_neuronet_vw(weigths + alpha*d, nn_))).^2);

      % Calculation the reference parameter
      Delta = 2*delta*(Jfunc(0)-Jfunc(alpha))/u^2
      
      if(Delta >= 0)
%           alpha
        weigths = weigths + alpha*d;
        nn_ = convert_w_to_neuronet_vw(weigths, nn_);
        g_i1 = -back_prop_batch_gradient(train_set, target, nn_);
        lambda_ = 0;
        success = 1;
        beta = max(0, (g_i1'*g_i)/(g_i'*g_i));
        if(mod(it, length(g_i)) == 0)
          d = g_i1;
        else
          beta = (g_i1'*g_i1 - g_i1'*g_i)/u;
          d = g_i1 + beta*d;
        end
        
        if(Delta >= 0.75)
          lambda = lambda/4;
        end
        
      else
        lambda_ = lambda;
        success = 0;
      end
      
      if(Delta < 0.25)
%           lambda
        lambda = lambda + (delta*(1-Delta)/(d'*d)); 
      end
      
      % Calculation MSE error
      mse_error = mean((target - neural_nete(train_set, nn_)).^2);
      it = it + 1;
      g_i1 = g_i;
      err_hist(it) = mse_error;
   
  end
  
end