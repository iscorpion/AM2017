function [Theta1 Theta2 acuracia] = redeNeural(X, y, lambda, input_layer_size, hidden_layer_size, num_labels, maxIterations)
m = size(X, 1);

%% Inicializando Theta

initial_Theta1 = inicializaPesosAleatorios(input_layer_size, hidden_layer_size);
initial_Theta2 = inicializaPesosAleatorios(hidden_layer_size, num_labels);

initial_rna_params = [initial_Theta1(:) ; initial_Theta2(:)];

%% Treinamento
options = optimset('MaxIter', maxIterations);

funcaoCusto = @(p) rnaCusto(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

[rna_params, cost] = fmincg(funcaoCusto, initial_rna_params, options);

Theta1 = reshape(rna_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(rna_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));



%% Predicao

pred = RN_predicao(Theta1, Theta2, X);

acuracia = mean(double(pred == y)) * 100;

end