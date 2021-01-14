function bias = computeBias(alphas, y, G, C)
% Computes the bias of the decision function
n = length(y);
SVs = [];
for i = 1:n
    if ((0 < alphas(i)) && (alphas(i) < C))
        SVs = [SVs i];
    end
end

numSVs = length(SVs);
bias = 0;
for i = SVs
    cur = y(i);
    for j= SVs
        cur = cur - alphas(j)*y(j)*G(i, j);
    end
    bias = bias + cur;
end
bias = (1/numSVs)*bias;


end

