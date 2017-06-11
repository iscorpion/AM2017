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

[X_train, Y_train, X_test, Y_test] = holdout(X, Y, 0.7);

fprintf('Dados carregados e processados com sucesso!\n\n')



%% ===================================================== %%
%% Menu Principal:
%% Opcoes:
%%    1 - Teste de parametros
%%    2 - Classificacao dos Dados


%% ===================================================== %%
%% Opcao 1 - Teste de Parametros
%%    Submenu 1 - Escolha de algoritmo para testar
%%        > knn (valores de K)
%%        > regressao logistica (lambda)
%%        > redes neurais
%%        > SVM
%% 

%% ===================================================== %%
%% Opcao 2 - Classificacao dos Dados
%%    Submenu 2 - Escolha um algoritmo de classificacao
%%        > knn (valores de K)
%%        > regressao logistica (lambda)
%%        > redes neurais
%%        > SVM
%%