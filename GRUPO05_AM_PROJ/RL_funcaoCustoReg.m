function [J, grad] = RL_funcaoCustoReg(theta, X, y, lambda)
%FUNCAOCUSTOREG Calcula o custo da regressao logistica com regularizacao
%   J = FUNCAOCUSTOREG(theta, X, y, lambda) calcula o custo de usar theta 
%   como parametros da regressao logistica para ajustar os dados de X e y 

% Initializa algumas variaveis uteis
m = length(y); % numero de exemplos de treinamento

% Voce precisa retornar as seguintes variaveis corretamente
J = 0;
grad = zeros(size(theta));

% ====================== ESCREVA O SEU CODIGO AQUI ======================
% Instrucoes: Calcule o custo de uma escolha particular de theta.
%             Voce precisa armazenar o valor do custo em J.
%             Calcule as derivadas parciais e encontre o valor do gradiente
%             para o custo com relacao ao parametro theta
%
% Obs: grad deve ter a mesma dimensao de theta
%

h = sigmoid(X*theta);

J0 = (1/m)*sum(-y' * log(h) - (1 - y')*log(1 - h));

J = J0 + lambda/(2*m)*sum(theta(2:end).^2);

Oj= [0, theta(2:end)']; % nao regulariza o theta 0

grad = ( sum((h- y).*X)/m ) + (lambda/m)*Oj;


% =============================================================

end
