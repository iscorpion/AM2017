function escreverResultado(filename, res)
fid = fopen(['resultado/' filename], 'w');
fprintf(fid, 'ID,TARGET\n');
fclose(fid) ;
dlmwrite(['resultado/' filename], res(:,:), '-append', 'precision', 20);
end