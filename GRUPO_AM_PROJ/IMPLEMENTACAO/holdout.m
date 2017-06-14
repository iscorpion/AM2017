function [X_train, Y_train, X_test, Y_test] = holdout(X, Y, S)
% HOLDOUT aplica uma divisao das amostras de X e Y em conjuntos
% de teste e de treinamento. A escolha de amostra eh aleatoria,
% e X_train e X_test sao subconjuntos disjuntos de X.

% Parametros:
%   X: conjunto de dados original
%   Y: conjunto de atributos alvo 
%   S: numero [0,1] indicando a divisao entre conjunto de treino 
%   e conjunto de teste. Ex.: S = 0.8 = 80% treinamento, 20% teste.


% Tamanho do conjunto de dados (linhas)
num_points = size(X,1);
% Define o ponto de divisao com base no parametro S
split = round(num_points*S);
% Cria uma sequencia aleatoria representando os indices de X
%seq = randperm(num_points);

% Divisao de X e Y em conjunto de treinamento
X_train = X(1:split,:);
Y_train = Y(1:split);

% Divisao de X e Y em conjunto de teste
X_test = X(split+1:end,:);
Y_test = Y(split+1:end);

end
