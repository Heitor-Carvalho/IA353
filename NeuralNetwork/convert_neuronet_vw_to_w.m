function [weigths] = convert_neuronet_vw_to_w(nn)
  
  mid_layer_weigths_number = (nn.in_sz+1)*nn.mid_sz;
  output_layer_weigths_number = (nn.mid_sz + 1)*nn.out_sz;
  weitghs_number = mid_layer_weigths_number + output_layer_weigths_number;

  weigths = zeros(weitghs_number, 1);

  weigths(1:mid_layer_weigths_number) = nn.v(:); 
  weigths(mid_layer_weigths_number+1:end) = nn.w(:);

end