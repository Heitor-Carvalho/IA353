function [opt_x, opt] = golden_search(min_interval, max_interval, func, stop_err)
  % Add description (min_interval, max_interval, func, stop_err) - seach by 
  % the minimum in the given interval   
  % Inputs:
  % min_interval : minimun value for seach interval
  % max_interval : maximun value for seach interval
  % func         : function to be minimized
  % stop_err     : minimum step progress to stop the search
  %
  % Outputs:
  % opt_x        : optimun point found
  % opt          : optimun function value
  
  x1 = min_interval;
  x2 = max_interval;
  while(abs(x1-x2) > stop_err)
    if(func(x1) < func(x2))
      x2 = (x1+x2)/2;
    else
      x1 = (x1+x2)/2;
    end
  end
  opt_x = x1;
  opt = func(x1);

end
