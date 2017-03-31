function [nn] = convert_w_to_neuronet_vw(weigths, nn)

  mid_layer_weigths_number = (nn.in_sz+1)*nn.mid_sz;
  output_layer_weigths_number = (nn.mid_sz + 1)*nn.out_sz;
  weitghs_number = mid_layer_weigths_number + output_layer_weigths_number;

  nn.v = reshape(weigths(1:mid_layer_weigths_number), nn.in_sz+1, nn.mid_sz);
  nn.w = weigths(mid_layer_weigths_number+1:end)';

end