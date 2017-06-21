%06/04/2017
%Equaliza��o supervisionada
%Entrada: canal - vetor com os coeficientes que definem a fun��o de transf. do canal
%               - Ex.: H(z) = 1.6 + z^{-1} --> canal = [1.6 1]
%         SNR   - rela��o sinal-ru�do em dB
%         K     - n�mero de amostras (atual e vers�es atrasadas) apresentadas na entrada da rede
%         atraso - estabelece o atraso de equaliza��o, i.e., qual a amostra passada da fonte que desejamos recuperar na sa�da do equalizador
%                - Ex.: atraso = 0 --> o sinal desejado � a amostra s(n)
%                - Ex.: atraso = 2 --> no instante n, o sinal desejado � s(n-2)
%         N_treinamento - n�mero total de amostras do conjunto de treinamento
%         N_teste - n�mero total de amostras do conjunto de teste
%Sa�da:  DADOS - estrutura contendo os dados para treinamento, valida��o e teste do equalizador
%Campos da estrutura:
%        DADOS.desejado_treinamento - vetor linha com as amostras desejadas para a sa�da da rede (1 x N_treinamento)
%        DADOS.entrada_treinamento - matriz (dimens�o K x N_treinamento) com as K amostras de entrada em cada instante de tempo de treinamento
%        DADOS.desejado_val - vetor linha com as amostras desejadas para a sa�da da rede (1 x N_val)
%        DADOS.entrada_val - matriz (dimens�o K x N_val) com as K amostras de entrada em cada instante de tempo de valida��o
%        DADOS.desejado_teste - vetor linha com as amostras desejadas para a sa�da da rede (1 x N_teste)
%        DADOS.entrada_teste - matriz (dimens�o K x N_teste) com as K amostras de entrada em cada instante de tempo de teste

function DADOS = gera_dados_equalizacao(canal,SNR,K,atraso,N_treinamento,N_val,N_teste)

%Vari�ncia do ru�do AWGN
sigma = sqrt(sum(canal.^2)*10^(-SNR/10));

%Treinamento

%amostras da fonte: s(n): s�mbolos +1 e -1 com probabilidade p = 0.5
s = sign(randn(1,N_treinamento+atraso));
%sinal recebido: r(n) = s(n)*h(n) + ru�do AWGN
rsig = filter(canal,1,s) + sigma*randn(1,N_treinamento+atraso);
%sinal desejado: � a pr�pria sequ�ncia de s�mbolos s(n)
DADOS.desejado_treinamento = s(1:N_treinamento);
%matriz de entrada do equalizador
if K > atraso + 1,
    DADOS.entrada_treinamento = toeplitz([rsig(atraso+1:-1:1)';zeros(K-(atraso+1),1)],rsig(atraso+1:end));
else
    DADOS.entrada_treinamento = toeplitz(rsig(atraso+1:-1:atraso-K+2),rsig(atraso+1:end));
end

%Valida��o

%amostras da fonte
sval = sign(randn(1,N_val+atraso));
%sinal recebido: r(n) = s(n)*h(n) + ru�do AWGN
rsig_val = filter(canal,1,sval) + sigma*randn(1,N_val+atraso);
%sinal desejado: � a pr�pria sequ�ncia de s�mbolos s(n)
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
%sinal recebido: r(n) = s(n)*h(n) + ru�do AWGN
rsigteste = filter(canal,1,steste) + sigma*randn(1,N_teste+atraso);
%sinal desejado: � a pr�pria sequ�ncia de s�mbolos s(n)
DADOS.desejado_teste = steste(1:N_teste);
%matriz de entrada do equalizador
if K > atraso + 1,
    DADOS.entrada_teste = toeplitz([rsigteste(atraso+1:-1:1)';zeros(K-(atraso+1),1)],rsigteste(atraso+1:end));
else
    DADOS.entrada_teste = toeplitz(rsigteste(atraso+1:-1:atraso-K+2),rsigteste(atraso+1:end));
end




