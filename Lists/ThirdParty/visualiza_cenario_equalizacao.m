%Rotina que exibe os estados do canal junto com os respectivos s�mbolos 
%(+1 ou -1), conforme o atraso de equaliza��o desejado
%Evidentemente, pressup�e que vamos trabalhar com um equalizador com duas
%entradas (K = 2)

function [C,S] = visualiza_cenario_equalizacao(canal,atraso)

%n�mero de coeficientes do canal
nc = length(canal);
%n�mero de entradas do equalizador
K = 2;
%obt�m os estados do canal
[C,S] = channel_states(canal,K);
%exibe os estados com as respectivas classes marcadas conforme o atraso
%selecionado
hold on;
for ii=1:2^(K+nc-1)
     if S(ii,atraso+1) == 1
        p1 = plot(C(1,ii),C(2,ii),'k+','MarkerSize',8,'LineWidth',2);
     else
        p2 = plot(C(1,ii),C(2,ii),'ko','MarkerSize',8,'LineWidth',2); 
     end
end
l1=['s(n-' num2str(atraso) ') = +1']; l2=['s(n-' num2str(atraso) ') = -1'];
legend([p1 p2],l1,l2);
xlabel('r(n)'); ylabel('r(n-1)'); grid;