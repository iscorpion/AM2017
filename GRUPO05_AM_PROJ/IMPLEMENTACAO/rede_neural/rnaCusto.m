function [J grad] = rnaCusto(nn_params, ...
                             input_layer_size, ...
                             hidden_layer_size, ...
                             num_labels, ...
                             X, y, lambda)
%RNACUSTO Implementa a funcao de custo para a rede neural com duas camadas
%voltada para tarefa de classificacao
%   [J grad] = RNACUSTO(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) calcula o custo e gradiente da rede neural. The
%   Os parametros da rede neural sao colocados no vetor nn_params
%   e precisam ser transformados de volta nas matrizes de peso.
%
%   input_layer_size - tamanho da camada de entrada
%   hidden_layer_size - tamanho da camada oculta
%   num_labels - numero de classes possiveis
%   lambda - parametro de regularizacao
%
%   O vetor grad de retorno contem todas as derivadas parciais
%   da rede neural.
%

% Extrai os parametros de nn_params e alimenta as variaveis Theta1 e Theta2.
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Definindo variaveis uteis
m = size(X, 1);
         
% As variaveis a seguir precisam ser retornadas corretamente
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== INSIRA SEU CODIGO AQUI ======================
% Instrucoes: Voce deve completar o codigo a partir daqui 
%               acompanhando os seguintes passos.
%
% (1): Lembre-se de transformar os rotulos Y em vetores com 10 posicoes,
%      onde tera zero em todas posicoes exceto na posicao do rotulo
%
% (2): Execute a etapa de feedforward e coloque o custo na variavel J.
%      Apos terminar, verifique se sua funcao de custo esta correta,
%      comparando com o custo calculado em ex05.m.
%
% (3): Implemente o algoritmo de backpropagation para calcular 
%      os gradientes e alimentar as variaveis Theta1_grad e Theta2_grad.
%      Ao terminar essa etapa, voce pode verificar se sua implementacao 
%      esta correta atraves usando a funcao verificaGradiente.
%
% (4): Implemente a regularização na função de custo e gradiente.
%

% size(Theta1): 25 x 401
% size(Theta2): 10 x 26

% Cost
Y = zeros(m, num_labels);
for i=1:m
  Y(i, y(i)) = 1;
endfor

plus1s = ones(m, 1);
a1 = [plus1s X]; % 5000 x 401 

z2 = a1 * Theta1'; % 5000 x 25
a2 = [plus1s 1./(1+exp(-z2))]; % 5000 x 26

z3 = a2 * Theta2'; % 5000 x 10
a3 = 1./(1+exp(-z3));

h = a3;

J = sum(sum(-Y .* log(h) - (1 - Y) .* log(1 - h)))/m;


% Regularization
Theta1WithoutBias = Theta1(:, [2:end]); % 25 x 400
Theta2WithoutBias = Theta2(:, [2:end]); % 10 x 25
J = J + (lambda / (2 * m)) * (sum(sum(Theta1WithoutBias .^ 2)) + sum(sum(Theta2WithoutBias .^ 2)));


% Backpropagation
for i=1:m
  % Step 1
  a1i = a1(i, :); % 1 x 401
  a2i = a2(i, :); % 1 x 26
  a3i = a3(i, :); % 1 x 10
  Yi = Y(i, :);   % 1 x 10
  
  % Step 2
  error3 = a3i - Yi; % 1 x 10
  
  % Step 3
  z2i = [1 z2(i, :)]; % 1 x 26
  error2 = error3 * Theta2 .* gradienteSigmoide(z2i); % 1 x 26
  
  % Step 4
  error2WithoutBias = error2(1, [2:end]); % 1 x 25
  
  Theta1_grad += error2WithoutBias' * a1i; % 25 x 401
  Theta2_grad += error3' * a2i;            % 10 x 26
endfor

% Step 5
Theta1_grad /= m; % 25 x 401
Theta2_grad /= m; % 10 x 26


% Regularization
Theta1_grad(:, [2:end]) += Theta1WithoutBias / m;
Theta2_grad(:, [2:end]) += Theta2WithoutBias / m;

% =========================================================================

% Junta os gradientes
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
