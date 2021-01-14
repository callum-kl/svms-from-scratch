function res = objectiveFunction(labels, alphas, G)
% Function to compute value of objective function in the dual formulation
% prod = labels.*alphas;
% res = -sum(alphas) + 0.5*sum(prod.'*prod*G, 'all');
res = 0;
N = length(labels);
for i = 1:N
    for j = 1:N
        res = res + labels(i)*labels(j)*alphas(i)*alphas(j)*G(i,j);
    end
end
res = 0.5*res - sum(alphas);
      
end

