%06/04/2017
%Equalização supervisionada
%Entrada: canal - vetor com os coeficientes que definem a função de transf. do canal
%               - Ex.: H(z) = 1.6 + z^{-1} --> canal = [1.6 1]
%         SNR   - relação sinal-ruído em dB
%         K     - número de amostras (atual e versões atrasadas) apresentadas na entrada da rede
%         atraso - estabelece o atraso de equalização, i.e., qual a amostra passada da fonte que desejamos recuperar na saída do equalizador
%                - Ex.: atraso = 0 --> o sinal desejado é a amostra s(n)
%                - Ex.: atraso = 2 --> no instante n, o sinal desejado é s(n-2)
%         N_treinamento - número total de amostras do conjunto de treinamento
%         N_teste - número total de amostras do conjunto de teste
%Saída:  DADOS - estrutura contendo os dados para treinamento, validação e teste do equalizador
%Campos da estrutura:
%        DADOS.desejado_treinamento - vetor linha com as amostras desejadas para a saída da rede (1 x N_treinamento)
%        DADOS.entrada_treinamento - matriz (dimensão K x N_treinamento) com as K amostras de entrada em cada instante de tempo de treinamento
%        DADOS.desejado_val - vetor linha com as amostras desejadas para a saída da rede (1 x N_val)
%        DADOS.entrada_val - matriz (dimensão K x N_val) com as K amostras de entrada em cada instante de tempo de validação
%        DADOS.desejado_teste - vetor linha com as amostras desejadas para a saída da rede (1 x N_teste)
%        DADOS.entrada_teste - matriz (dimensão K x N_teste) com as K amostras de entrada em cada instante de tempo de teste

function DADOS = gera_dados_equalizacao(canal,SNR,K,atraso,N_treinamento,N_val,N_teste)

%Variância do ruído AWGN
sigma = sqrt(sum(canal.^2)*10^(-SNR/10));

%Treinamento

%amostras da fonte: s(n): símbolos +1 e -1 com probabilidade p = 0.5
s = sign(randn(1,N_treinamento+atraso));
%sinal recebido: r(n) = s(n)*h(n) + ruído AWGN
rsig = filter(canal,1,s) + sigma*randn(1,N_treinamento+atraso);
%sinal desejado: é a própria sequência de símbolos s(n)
DADOS.desejado_treinamento = s(1:N_treinamento);
%matriz de entrada do equalizador
if K > atraso + 1,
    DADOS.entrada_treinamento = toeplitz([rsig(atraso+1:-1:1)';zeros(K-(atraso+1),1)],rsig(atraso+1:end));
else
    DADOS.entrada_treinamento = toeplitz(rsig(atraso+1:-1:atraso-K+2),rsig(atraso+1:end));
end

%Validação

%amostras da fonte
sval = sign(randn(1,N_val+atraso));
%sinal recebido: r(n) = s(n)*h(n) + ruído AWGN
rsig_val = filter(canal,1,sval) + sigma*randn(1,N_val+atraso);
%sinal desejado: é a própria sequência de símbolos s(n)
DADOS.desejado_val = sval(1:N_val);
%matriz de entrada do equalizador
if K > atraso + 1,
    DADOS.entrada_val = toeplitz([rsig_val(atraso+1:-1:1)';zeros(K-(atraso+1),1)],rsig_val(atraso+1:end));
else
    DADOS.entrada_val = toeplitz(rsig_val(atraso+1:-1:atraso-K+2),rsig_val(atraso+1:end));
end

%Teste

%amostras da fonte
steste = sign(randn(1,N_teste+atraso));
%sinal recebido: r(n) = s(n)*h(n) + ruído AWGN
rsigteste = filter(canal,1,steste) + sigma*randn(1,N_teste+atraso);
%sinal desejado: é a própria sequência de símbolos s(n)
DADOS.desejado_teste = steste(1:N_teste);
%matriz de entrada do equalizador
if K > atraso + 1,
    DADOS.entrada_teste = toeplitz([rsigteste(atraso+1:-1:1)';zeros(K-(atraso+1),1)],rsigteste(atraso+1:end));
else
    DADOS.entrada_teste = toeplitz(rsigteste(atraso+1:-1:atraso-K+2),rsigteste(atraso+1:end));
end




