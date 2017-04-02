function [nn_, err_hist, it] = back_prop_batch_jac(train_set, target, nn, train_par)
  % To do: add support to multiples outputs
  % To do: add function description
  % Levenberg-Marquart  
  nn_ = nn;
  mse_error = train_par.max_error;
  err_hist = zeros(1, train_par.max_it);
  it = 0;

  delta_w_past = 0;
  delta_v_past = 0;
  
  samples_sz = size(train_set, 2);

  while(mse_error >= train_par.max_error && ...
        it        < train_par.max_it)
   
    mse_error = 0;
  
    % Foward part - Neural network output
    [nn_out, mid_layer_func_out_bias, ~, mid_layer_func_in, in_bias] = neural_nete(train_set, nn_);  

    % Neuro network error
    error = target - nn_out;
    error = reshape(error, 1 , 1, samples_sz);
    keyboard    
    mse_error = mean(error.^2);
      
    % Output layer weights gradient
    derror_dw = -2*repmat(error, 1, nn_.mid_sz+1).*mid_layer_func_out_bias;
    
    % Middle layer weights gradient
    w = repmat(nn_.w(:, 2:end), 1, 1, samples_sz);
    
    
    derror_dv = -2*repmat(error, nn_.in_sz+1, nn_.mid_sz)                            .* ...
                   repmat(w, nn_.in_sz+1, 1)                                         .* ...
                   repmat(nn_.diff(mid_layer_func_in), nn_.in_sz+1, 1)               .* ...
                   repmat(in_bias, 1, nn_.mid_sz);

    Jv = reshape(derror_dv, (nn_.in_sz+1)*nn_.mid_sz, samples_sz)';
    Jw = reshape(derror_dw, nn_.mid_sz+1, samples_sz)';

    J = [Jv Jw];
    
    % Levenberg-Marquart Method    
%     while(1)
%         keyboard
      d2J = 2*pinv(J'*J + train_par.mu*eye(size(J,2)));
      d2J = 1;
      deltaW = d2J*J'*reshape(error, samples_sz, 1);

      weigths = convert_neuronet_vw_to_w(nn_);

      Jfunc = @(alpha) mean((target - neural_nete(train_set, convert_w_to_neuronet_vw(weigths - alpha*deltaW, nn_))).^2);

      alpha =  golden_search(0, 10, Jfunc, 1e-4)
    
      if(Jfunc(alpha) < mse_error)
        train_par.mu = max(train_par.mu*0.5, 1e-6);
    weigths = weigths - alpha*deltaW;
    nn_ = convert_w_to_neuronet_vw(weigths, nn_);
      else
        train_par.mu = min(train_par.mu*2, 1e3);
%         break
      end
        mu = train_par.mu
    
%     end
 

    it = it + 1;
%         mu = train_par.mu
    mse_error
    err_hist(it) = mse_error;

  end
  
end