%Rotina que exibe o mapeamento entrada-sa�da e a fronteira de decis�o de
%uma rede MLP aplicada ao problema de equaliza��o

%Entradas:
%   canal - vetor com os coeficientes do canal (1 x nc)
%   atraso - atraso de equaliza��o, i.e., s(n-atraso) dever� ser recuperado na sa�da da rede
%   W1 - matriz de pesos da camada de entrada da MLP - (N x K+1)
%       N - n�mero de neur�nios na camada intermedi�ria
%       K � o n�mero de entradas externas (no caso, para haver visualiza��o do mapeamento, K = 2)
%   W2 - matriz de pesos da camada de sa�da da MLP - (1 x N+1)
%       No caso geral, seria L x N+1, onde L � o n�mero de sa�das da rede
%   entrada - matriz com os dados de entrada do conjunto de treinamento (K x NT), onde NT denota o n�mero de padr�es (amostras) de entrada

function map_fronteira_mlp_equalizacao_rbf(canal,atraso,W1,W2,centers,sig,entrada)

%n�mero de coeficientes do canal
W1 = W1';
nc = length(canal);
%n�mero de entradas da rede = dimens�o dos estados do canal (subtrai o bias)
K = size(W1,2); K = K - 1; 
%obt�m os estados do canal
%keyboard()
[C,S] = channel_states(canal,K);
%Intervalo de an�lise do mapeamento
x0 = (min(min(C))-1:.1:max(max(C))+1); MM = max(size(x0)); x1 = x0;
%MAPEAMENTO ENTRADA-SA�DA DA REDE
mlp = zeros(MM,MM);
for jj = 1:MM
    for ii = 1:MM
        xx = [1; x0(jj); x1(ii);]; %[r(k) r(k-1) bias]
%        keyboard()
        %sa�da da rede para a entrada xx
        xv = repmat(xx, 1, size(W1,1));
        xv(2:3,:) = xv(2:3,:) - centers;
        mlp(ii,jj) = W2*[1; exp(-(sqrt(sum(W1'.*xv.^2, 1))').^2/sig)];
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
%VISUALIZA��O DO CEN�RIO TRATADO E FRONTEIRA DE DECIS�O
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