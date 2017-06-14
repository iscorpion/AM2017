%% Universidade Federal de Sao Carlos - UFSCar, Sorocaba
%
%  Disciplina: Aprendizado de M�quina
%  Prof. Tiago A. Almeida
%
%  Trabalho 01 - Censo 1994 - Quem Ganha Mais de 50 Mil?
%
%  Grupo 05:
%  ----------
%  Celso Martins De Araujo Filho
%
%  Vinnícius Ferreira da Silva
%
%  Renan Ferreira de Almeida
%  ----------
%
%

%% Inicializacao
clear ; close all; clc

addpath('rede_neural');

%% ===================================================== %%
%% Carregamento dos dados

fprintf('Carregando dados...\n')
data = csvread("adult_processed.csv");



%% ===================================================== %%
%% Preprocessamento dos dados

fprintf('Processando dados...\n')

% Remove primeira linha (cabeçalho)
data(1, :) = [];

% Remove coluna redundante "education"
data(:, 4) = [];

X = data(:, 1:end-1);
Y = data(:, end);
Y += 1; % Muda classes de 0 e 1 para 1 e 2
clear data;
% Normalizacao dos dados com media 0 e desvio 1
%X = X(1:150,:);
%Y = Y(1:150);
[X_norm, ~, ~] = normalizar(X);

fprintf('Dados carregados e processados com sucesso!\n\n')



%% ===================================================== %%
%% Menu Principal:
%% Opcoes:
%%    1 - Teste de parametros
%%    2 - Classificacao dos Dados
input1 = input("Selecione uma opcao:\n1 - Teste de Parametros\n2 - Classificacao dos Dados\n0 - Sair\n\n");
while(input1 ~= 0)

  %% ===================================================== %%
  %% Opcao 1 - Teste de Parametros
  %%    Submenu 1 - Escolha de algoritmo para testar
  %%        > knn (valores de K)
  %%        > regressao logistica (lambda)
  %%        > redes neurais
  %%        > SVM
  %% 
  if(input1 == 1)
    % Holdout nos dados    
    [X_train, Y_train, X_test, Y_test] = holdout(X_norm, Y, 0.7);
    
    input2 = input("\nEscolha um algoritmo:\n1 - k-NN\n2 - Regressao Logistica\n3 - Redes Neurais\n4 - SVM\n\n");
    
    if(input2 == 1)
      fprintf("KNN escolhido\nTestando com valores K de 1 ate 151\n\n");
      
      K_range = 1:2:151;
      knn_historico = zeros(size(K_range,2),1);
      J = 1;
      for it = K_range
        [y, ~] = knn(X_test, X_train, Y_train, it);
                
        knn_historico(J) = sum(y == Y_test)/size(y,1);
        J = J+1;
      endfor
      
      fprintf("Taxa de Acerto KNN\n");
      for C = 1:size(K_range,2)
        fprintf("%.2f%% | K = %d\n", (knn_historico(C)*100), K_range(C));
      endfor
      [knn_max_hist,idx] = max(knn_historico);
      fprintf("\nValor de K escolhido: %d (%.2f%%)\n\n", K_range(idx), knn_max_hist*100);
            
   elseif(input2 == 2)
      fprintf("Taxa de Acerto Regressao Logistica\n");
      
      rl_historico = zeros(20,1);
      it = 1;
      RL_range = -10:10;
      for I = RL_range
        [m, n] = size(X);

        X_temp = atributosPolinomiais(X_train(:,1), X_train(:,2));
        
        theta_inicial = zeros(size(X_temp, 2), 1);
        
        lambda = 10^I;

        [custo, grad] = RL_funcaoCustoReg(theta_inicial, X_temp, Y_train, lambda);
              
        opcoes = optimset('GradObj', 'on', 'MaxIter', 400);
        
        [theta, J, exit_flag] = ...
          fminunc(@(t)(RL_funcaoCustoReg(t, X_temp, Y_train, lambda)), theta_inicial, opcoes);
                    
        p = RL_predicao(theta, X_temp);
    
        %fprintf('Acuracia na base de treinamento: %f\n', mean(double(p == Y_train)) * 100);
        
        X_temp = atributosPolinomiais(X_test(:,1), X_test(:,2));
        classe = RL_predicao(theta, X_temp);
        
        fprintf('%.2f%% (lambda = %d)\n', mean(double(classe == Y_test)) * 100, lambda);
        
        rl_historico(it) = mean(double(classe == Y_test)) * 100;
        it = it+1;
        
      endfor
      [rl_historico_max idx] = max(rl_historico);
      fprintf("Melhor valor de lambda: %d (%.2f%%)\n\n", 10^RL_range(idx), rl_historico_max);
      
    elseif(input2 == 3)
      fprintf("Voce escolheu Redes Neurais\n");
      input_layer_size  = size(X, 2);  % Numero de colunas de X
      num_labels = 2;
      
      initial_hidden_layer_size = ceil((input_layer_size + 2)/2); % Como base, um valor intermediário entre o número de entradas e saídas
     
      maxIterations = 50;

      acuracias = zeros(10, 1);
      acuracias_test = zeros(10, 1);
      
      % Teste de valores de lambda de 1 até 10
      for i = 1:10
        lambda = i;
        [Theta1 Theta2 acuracia] = redeNeural(X_train, Y_train, lambda, input_layer_size, initial_hidden_layer_size, num_labels, maxIterations);
        acuracias(i, 1) = acuracia;
        
        pred = RN_predicao(Theta1, Theta2, X_test);
        acuracia_test = mean(double(pred == Y_test)) * 100;
        
        acuracias_test(i, 1) = acuracia_test;
      endfor
     
      fprintf('\nAcuracias na base de treinamento por Lambda\n')
      for i = 1:10
        fprintf('%d : %.2f%%\n', i, acuracias(i))
      endfor

      most_accurate_lambda = 1;
      
      fprintf('\nAcuracias na base de teste por Lambda\n')
      for i = 1:10
        fprintf('%d : %.2f%%\n', i, acuracias_test(i))
        if acuracias_test(i) > acuracias_test(most_accurate_lambda)
          most_accurate_lambda = i;
        endif
      endfor
      
      lambda = most_accurate_lambda;
      fprintf('\nO melhor Lambda foi %d\n', lambda)
      
      
      acuracias = zeros(6, 1);
      acuracias_test = zeros(6, 1);
           
      % Teste de valores de hidden_layer_size usando o lambda encontrado
      min_size = initial_hidden_layer_size - 2;
      for i = 1:6
        hidden_layer_size = i - 1 + min_size;
        [Theta1 Theta2 acuracia] = redeNeural(X_train, Y_train, lambda, input_layer_size, hidden_layer_size, num_labels, maxIterations);
        acuracias(i, 1) = acuracia;
        
        pred = RN_predicao(Theta1, Theta2, X_test);
        acuracia_test = mean(double(pred == Y_test)) * 100;
        
        acuracias_test(i, 1) = acuracia_test;
      endfor
      
      fprintf('\nAcuracias na base de treinamento por tamanho da camada oculta\n')
      for i = 1:6
        fprintf('%d : %.2f%%\n', i - 1 + min_size, acuracias(i))
      endfor

      most_accurate_size = 1;
      
      fprintf('\nAcuracias na base de teste por tamanho da camada oculta\n')
      for i = 1:6
        fprintf('%d : %.2f%%\n', i - 1 + min_size, acuracias_test(i))
        if acuracias_test(i) > acuracias_test(most_accurate_size)
          most_accurate_size = i;
        endif
      endfor
      
      fprintf('\nO melhor tamanho da camada oculta foi %d\n', most_accurate_size)
    elseif(input2 == 4)
      fprintf("Voce escolheu SVM\n");
    endif
    
  %% ===================================================== %%
  %% Opcao 2 - Classificacao dos Dados
  %%    Submenu 2 - Escolha um algoritmo de classificacao
  %%        > knn (valores de K)
  %%        > regressao logistica (lambda)
  %%        > redes neurais
  %%        > SVM
  %%    
        
  elseif(input1 == 2)
    input2 = input("\nEscolha um algoritmo:\n1 - k-NN\n2 - Regressao Logistica\n3 - Redes Neurais\n4 - SVM\n\n");
    
    if(input2 == 1)
      fprintf("Voce escolheu KNN\n");
    elseif(input2 == 2)
      
      fprintf("\nTaxa de Acerto Regressao Logistica\n");
      rl_historico = zeros(5,1);
      
      it = 1;
      for I = 1:5
      
        % K-fold CV com K = 5
        K = 5;
        [X_train Y_train X_test Y_test] = kfold(X, Y, K, I);

        X_temp = atributosPolinomiais(X_train(:,1), X_train(:,2));
        
        theta_inicial = zeros(size(X_temp, 2), 1);
        
        lambda = 10^4;

        [custo, grad] = RL_funcaoCustoReg(theta_inicial, X_temp, Y_train, lambda);
              
        opcoes = optimset('GradObj', 'on', 'MaxIter', 400);
        
        [theta, J, exit_flag] = ...
          fminunc(@(t)(RL_funcaoCustoReg(t, X_temp, Y_train, lambda)), theta_inicial, opcoes);
                    
        p = RL_predicao(theta, X_temp);
    
        %fprintf('Acuracia na base de treinamento: %f\n', mean(double(p == Y_train)) * 100);
        
        X_temp = atributosPolinomiais(X_test(:,1), X_test(:,2));
        classe = RL_predicao(theta, X_temp);
        
        fprintf('%.2f%% (Particao = %d)\n', mean(double(classe == Y_test)) * 100, I);
        
        rl_historico(it) = mean(double(classe == Y_test)) * 100;
        it = it+1;
        
      endfor
      fprintf("Taxa de acerto media: %.2f%%\n\n", mean(rl_historico));
     
    elseif(input2 == 3)
      fprintf("\nTaxa de Acertos Rede Neural\n");
 
      rede_neural_historico = zeros(5,1);
 
      it = 1;
      for I = 1:5
        % K-fold CV com K = 5
        K = 5;
        [X_train Y_train X_test Y_test] = kfold(X, Y, K, I);
 
        input_layer_size  = size(X, 2);  % Numero de colunas de X
        maxIterations = 50;
        num_labels = 2;

        lambda = 2;
        hidden_layer_size = 6;

        [Theta1 Theta2 acuracia] = redeNeural(X_train, Y_train, lambda, input_layer_size, hidden_layer_size, num_labels, maxIterations);
      
        fprintf('%.2f%% (Particao = %d)\n', acuracia, I);

        rede_neural_historico(it) = acuracia;

        it = it+1;
      endfor
 
      fprintf("Taxa de acerto media: %.2f%%\n\n", mean(rede_neural_historico));
 
    elseif(input2 == 4)
      fprintf("Voce escolheu SVM\n");  
    endif
      
  endif
  
  input1 = input("\nSelecione uma opcao:\n1 - Teste de Parametros\n2 - Classificacao dos Dados\n0 - Sair\n\n");  
endwhile
