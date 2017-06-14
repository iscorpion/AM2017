function [X_train Y_train X_test Y_test] = kfold(X, Y, K, I)
% KFOLD particiona as amostras de X e Y em conjuntos
% de teste e de treinamento. A escolha de amostra eh feita com
% base em K. Cada chamada dessa funcao retorna a i-esima particao

% Parametros:
%   X(mxn): conjunto de dados original
%   Y(mx1): conjunto de atributos alvo 
%   K(1x1): numero de folds. Valores comuns: 10 e 5
%   I(1x1): iteracao atual (I = 1:K)
%
% Usagem:
%   Essa funcao devera ser chamada dentro de um laco, que executara
%   ate K vezes. Ex.: X(100x2), K = 5, I = 1
%     X_train(80x2) - 21:100
%     X_test(20x2) - 1:20

% Numero total de amostras
num = size(X, 1);
% Define o ponto de divisao com base em K
split = round(num/K);

if(I == 1)
  % Divisao de X e Y em conjunto de teste
  X_test = X(1:split,:);
  Y_test = Y(1:split,:);
  
  % Divisao de X e Y em conjunto de treinamento
  X_train = X(split+1:end,:);
  Y_train = Y(split+1:end,:);
  
else
  % Divisao de X e Y em conjunto de teste
  J = I - 1;
  X_test = X(split*J+1:(split*J+split), :);
  Y_test = Y(split*J+1:(split*J+split), :);
  
  % Divisao de X e Y em conjunto de treinamento
  X_train = [X(1:split*J,:); X((split*J+split)+1:end, :)];
  Y_train = [Y(1:split*J,:); Y((split*J+split)+1:end, :)];  
end

end