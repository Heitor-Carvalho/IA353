addpath('./ThirdParty/')

clear all

rng(33);
data_generation = 1;
noise_graph = 1;

% Case 1
h_case1 = [0.9 1]                           % Channel
d_case1 = 0;                                % Delay

% Case 1
h_case2 = [1 0.5 0.2]                       % Channel
d_case2 = 2;                              % Delay


snr = 15;
n_traing = 2000;
n_validation = 5000;
n_test = 5000;

% Data generation
if(data_generation)
  channel_samples_case1 = gera_dados_equalizacao(h_case1, snr, 2, d_case1, n_traing, n_validation, n_test);
  channel_samples_case2 = gera_dados_equalizacao(h_case2, snr, 2, d_case2, n_traing, n_validation, n_test);
end

% Item (a)
visualiza_cenario_equalizacao(h_case1, d_case1);
visualiza_cenario_equalizacao(h_case2, d_case2);

if(noise_graph)
  % Generate noise graphs using the training data

  % Case 1
  plus_one_idx = channel_samples_case1.desejado_treinamento(1,:) > 0;
  plot(channel_samples_case1.entrada_treinamento(1, plus_one_idx), channel_samples_case1.entrada_treinamento(2, plus_one_idx), '+r')
  hold on
  plot(channel_samples_case1.entrada_treinamento(1, ~plus_one_idx), channel_samples_case1.entrada_treinamento(2, ~plus_one_idx), 'ob')
  
  % Case 2
  plus_one_idx = channel_samples_case2.desejado_treinamento(1,:) > 0;
  plot(channel_samples_case2.entrada_treinamento(1, plus_one_idx), channel_samples_case2.entrada_treinamento(2, plus_one_idx), '+r')
  hold on
  plot(channel_samples_case2.entrada_treinamento(1, ~plus_one_idx), channel_samples_case2.entrada_treinamento(2, ~plus_one_idx), 'ob')
end
