function Y = mapcol(X, f)

cells = arrayfun(@(i) f(X(:,i)), 1:size(X)(2), 'UniformOutput', false);
Y = cell2mat(cells);

end