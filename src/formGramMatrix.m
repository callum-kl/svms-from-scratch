function G = formGramMatrix(X, type, param)
% Forms the Kernel Gram matrix with inputs x1 and x2 and
% using kernel type 'type'
n = length(X);
G = zeros(n);

switch type
    case 'linear'
        G = X*X.';
    case 'poly'
        G = (X*X.').^param;
    case 'gaussian'
        K = @(x,y) exp(-(1/param)*norm(x-y)^2);
        for i = 1:n
            for j = 1:n
                G(i,j) = K(X(i,:), X(j,:)); 
            end
        end
end
end

