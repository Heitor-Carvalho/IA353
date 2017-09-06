function kernel = poly_kernel(x, y, gamma, coef, degree)

  gamma = gamma/size(x,2);
  kernel = (x*y'*gamma + coef).^degree;

end
