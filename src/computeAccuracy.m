function acc = computeAccuracy(labels, alphas, bias, G)

classify = @(i) sign(decisionFunction(i, labels, alphas, bias, G));
acc = 0;
N = length(labels);
for i = 1:N
    pred = classify(i);
    if pred == labels(i)
        acc = acc + 1;
    end
end
acc = acc/N;
end

