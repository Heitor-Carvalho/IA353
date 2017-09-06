function kernel = poly_kernel(x, y, gamma, coef)

  gamma = gamma/size(x,2);
  kernel = tanh(x*y'*gamma + coef);

end
