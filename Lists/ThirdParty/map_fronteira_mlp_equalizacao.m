%Rotina que exibe o mapeamento entrada-saída e a fronteira de decisão de
%uma rede MLP aplicada ao problema de equalização

%Entradas:
%   canal - vetor com os coeficientes do canal (1 x nc)
%   atraso - atraso de equalização, i.e., s(n-atraso) deverá ser recuperado na saída da rede
%   W1 - matriz de pesos da camada de entrada da MLP - (N x K+1)
%       N - número de neurônios na camada intermediária
%       K é o número de entradas externas (no caso, para haver visualização do mapeamento, K = 2)
%   W2 - matriz de pesos da camada de saída da MLP - (1 x N+1)
%       No caso geral, seria L x N+1, onde L é o número de saídas da rede
%   entrada - matriz com os dados de entrada do conjunto de treinamento (K x NT), onde NT denota o número de padrões (amostras) de entrada

function map_fronteira_mlp_equalizacao(canal,atraso,W1,W2,entrada)

%número de coeficientes do canal
nc = length(canal);
%número de entradas da rede = dimensão dos estados do canal (subtrai o bias)
K = size(W1,2); K = K - 1; 
%obtém os estados do canal
[C,S] = channel_states(canal,K);
%Intervalo de análise do mapeamento
x0 = (min(min(C))-1:.01:max(max(C))+1); MM = max(size(x0)); x1 = x0;
%MAPEAMENTO ENTRADA-SAÍDA DA REDE
mlp = zeros(MM,MM);
for jj = 1:MM
    for ii = 1:MM
        xx = [x0(jj); x1(ii); 1]; %[r(k) r(k-1) bias]
        %saída da rede para a entrada xx
        mlp(ii,jj) = W2*[tanh(W1*xx);1];
    end
end
figure; mesh(x0,x1,mlp); xlabel('r(n)'); ylabel('r(n-1)'); zlabel('y_{MLP}(n)'); colormap jet;
hold on;
for ii=1:2^(K+nc-1)
     if S(ii,atraso+1) == 1
        plot3(C(1,ii),C(2,ii),1,'k+','MarkerSize',8,'LineWidth',2);
     else
        plot3(C(1,ii),C(2,ii),-1,'k*','MarkerSize',8,'LineWidth',2); 
     end
end
%VISUALIZAÇÃO DO CENÁRIO TRATADO E FRONTEIRA DE DECISÃO
figure; 
plot(entrada(1,:),entrada(2,:),'r.');
hold on;  
for ii=1:2^(K+nc-1)
     if S(ii,atraso+1) == 1
        p1 = plot(C(1,ii),C(2,ii),'k+','MarkerSize',8,'LineWidth',2);
     else
        p2 = plot(C(1,ii),C(2,ii),'ko','MarkerSize',8,'LineWidth',2); 
     end
end
legend([p1 p2],'s(n-d) = +1','s(n-d) = -1');
xlabel('r(n)'); ylabel('r(n-1)'); contour(x0,x1,mlp,[0 0],'k');


