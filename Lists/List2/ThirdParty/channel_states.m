%Rotina que determina os estados de um canal para a modulação BPSK
%Entrada: h - coeficientes do canal (H(z))
%         m - dimensão dos estados

function [C,S1] = channel_states(h,m)

%comprimento do canal
nc = length(h);

%matriz de convolução
H = toeplitz([h(1) zeros(m-1)],[h zeros(m-1)]);          %       k k-1    k-(m+nc-1)+1
%possíveis combinações de m+nc-1 valores de entrada      %S1 = [-1 -1 ... -1
N = m+nc-1; S1 = rem(floor([0:2^N-1]' * pow2(1-N:0)),2); %      -1 -1 ...  1 
%substitui os zeros por -1                               %       .  .      .
S1(S1 == 0) = -1;
%põe na ordem correta
S1 = -S1;

%estados do canal
C = H*S1';

