function res = decisionFunction(index, labels, alphas, bias, G)
res = sum(labels'.*alphas.*G(index,:)) - bias;
end

