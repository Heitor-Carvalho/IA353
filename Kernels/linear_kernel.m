function kernel = poly_kernel(x, y)

  gamma = gamma/size(x,2);
  kernel = x*y';

end
