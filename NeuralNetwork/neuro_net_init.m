function [nn] = neuro_net_init(nn)
  % neuro_net_init(nn) - Initialze a neural netwok struct
  % adding fields informing the network input size, middle layer size 
  % and number of output. Also, check consistent betwen middle layer 
  % size and number of outputs weigths.
  % Inputs:
  %   nn.v    : middle layer weights
  %   nn.w    : output layer weights

  nn.in_sz = size(nn.v, 1)-1;
  nn.mid_sz = size(nn.v, 2);

  if(isfield(nn, 'w'))
    nn.out_sz = size(nn.w, 1);
  
    % Checking number of output weigths
    assert(nn.mid_sz == size(nn.w, 2)-1, 'Unexpected number of collumn for w, should be %d', nn.mid_sz+1);
  end
  

end
