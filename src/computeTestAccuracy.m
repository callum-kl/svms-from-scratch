function acc = computeTestAccuracy(ytest, Xtest, ytrain, Xtrain, alphas, bias, K)

acc = 0;
for i = 1:length(ytest)
    yhat = 0;
    for j = 1:length(alphas)
        yhat = yhat + ytrain(j)*alphas(j)*K(Xtrain(j,:), Xtest(i,:)); 
    end
    yhat = sign(yhat + bias);
    if yhat == ytest(i)
        acc = acc + 1;
    end
end
acc = acc/length(ytest);
end

