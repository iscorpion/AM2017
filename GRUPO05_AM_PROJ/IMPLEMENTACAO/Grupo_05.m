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
addpath('libsvm-3.22/');
addpath('libsvm-3.22/matlab');
addpath('libsvm-3.22/window');
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
    
    t = clock;
    fprintf("\nTime: %02d:%02d\n", t(4), t(5));
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
      
      t = clock;
      fprintf("\nTime: %02d:%02d\n", t(4), t(5));  
      
    elseif(input2 == 2)
      fprintf("Voce escolheu Regressao Logistica\n");
    elseif(input2 == 3)
      fprintf("Voce escolheu Redes Neurais\n");
    elseif(input2 == 4)
      fprintf("Voce escolheu SVM\n\n");
      
      better_accuracy = 0.0;
      better_c = 0.0;
      better_gama = 0.0;
      fprintf("Taxa de Acerto SVM\n");
      fflush(stdout);
      
      for i = 1:1
        c = power(2,i);
        for j = 1:1
          gama = power(2,j);
          libsvm_options = sprintf('-c %.2f -g %0.f -t 2', c, gama);
          SVMStruct = svmtrain(Y_train, X_train, libsvm_options);
          [labels, accuracy, prob] = svmpredict(Y_test, X_test, SVMStruct);
          
          fprintf('\n%f% - Valor de c = %f e gama = %f\n', accuracy(1), c, gama);
          fflush(stdout);
          if (better_accuracy <= accuracy(1))
            better_accuracy = accuracy(1);
            better_c = c;
            better_gama = gama;
          endif
        endfor
      endfor
      fprintf("\nMelhores parametros testados foram c = %f e gama = %f, com acuracia de %f%%.\n", better_c, better_gama, better_accuracy);
      fflush(stdout);
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
      fprintf("Voce escolheu Regressao Logistica\n");
    elseif(input2 == 3)
      fprintf("Voce escolheu Redes Neurais\n");
      input_layer_size  = size(X, 2);  % Numero de colunas de X
      hidden_layer_size = 8;   % 8 neuronios na camada oculta
      num_labels = 2;

      lambda = 3;
      maxIterations = 50;

      [Theta1 Theta2 acuracia] = redeNeural(X, Y, lambda, input_layer_size, hidden_layer_size, num_labels, maxIterations)  
      
    elseif(input2 == 4)
      fprintf("Voce escolheu SVM\n");  
      fprintf('Iniciando treinamento SVM\n');
      [X_train, Y_train, X_test, Y_test] = holdout(X_norm, Y, 0.7);
      %% Segundo os testes que podem ser replicados pelas opçÕes 1 e 4 destes trabalho,
      %% os melhores parametros para esse problema são c=2 e gama = 0.5
      SVMStruct = svmtrain(Y_train,X_train,'-c 2 -g 0.5 -t 2');
      fprintf('Iniciando prediÁ„o SVM\n');
      [labels, accuracy, prob] = svmpredict(Y_test, X_test, SVMStruct); % run the SVM model on the test data
      fprintf('\nAcuracia no conjunto de treinamento: %f\n', accuracy);
      
      %%fprintf('SVM finalizado. Salvando resultados em svm.csv\n');
      %%escreverResultado(['svm.csv'], [Y_test labels]); %Escreve o resultado do knn em um arquivo csv
      visualizarDados(X_test, labels);
    endif
      
  endif
  
  input1 = input("\nSelecione uma opcao:\n1 - Teste de Parametros\n2 - Classificacao dos Dados\n0 - Sair\n\n");  
endwhile
