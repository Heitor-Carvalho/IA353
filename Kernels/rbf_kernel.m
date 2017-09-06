function kernel = rbf_kernel(x, y, gamma)

  gamma = gamma/size(x,2);
  dist_matrix = pdist2(x, y);
  
  kernel = exp(-gamma*dist_matrix.^2);

end
