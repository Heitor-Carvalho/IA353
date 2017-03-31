function [opt_x, opt] = golden_search(max_interval, min_interval, func, stop_err)
  % golden_search - seach by the minimum in the given interval   
  %
  % max_interval - maximun seach interval
  % min_interval - minimun seach interval
  % func- function to be minimized
  % stop_err - minimum step progress to stop search

  x1 = max_interval;
  x2 = min_interval;
  
  while(abs(x1-x2) > stop_err)
    x1 = max_interval - 0.61803*(max_interval-min_interval);
    x2 = min_interval + 0.61803*(max_interval-min_interval);
  
    if(func(x1) < func(x2))
      max_interval = x2;
    else
      min_interval = x1;
    end
  end
  
  opt_x = x1;
  opt = func(x1);

end