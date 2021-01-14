function [res, alphas, errorCache, bias] = examineExample(i2, labels, alphas, errorCache, C, bias, eps, tol, G)

alph2 = alphas(i2);
y2 = labels(i2);

res = 0;

% If bound example recompute error value
if (alph2 > tol && alph2 < C - tol)
    E2 = errorCache(i2);
else
    E2 = decisionFunction(i2, labels, alphas, bias, G) - y2;
end

r2 = y2*E2;

if((r2 < -tol && alph2 < C) || (r2 > tol && alph2 > 0))
    % possibly errors?
    nonBoundIndices = find((alphas > tol) & (alphas < C-tol));
    % If there are non-bound examples
    if ~isempty(nonBoundIndices)
       % Find maximal violating working pair index i2
       if E2 > 0
           [~, i1] = max(errorCache);
           [res, alphas, errorCache, bias] = takeStep(i1, i2, labels, alphas, errorCache, G, bias, C, eps, tol, E2);
           if res == 1
               return
           end
           
       elseif E2 < 0
          
           [~,i1] = min(errorCache);
           [res, alphas, errorCache, bias] = takeStep(i1, i2, labels, alphas, errorCache, G, bias, C, eps, tol, E2);
           if res == 1
               return;
           end
       end
       % Or pick any pair i2 randomly if no progress made
       randIndices = randperm(length(nonBoundIndices));
       for index = randIndices
           i1 = nonBoundIndices(index);
           [res, alphas, errorCache, bias] = takeStep(i1, i2, labels, alphas, errorCache, G, bias, C, eps, tol, E2);
           if res == 1
               return
           end
       end
    end
    
    % Else iterate through all examples
    for i1 = 1:length(alphas)
        [res, alphas, errorCache, bias] = takeStep(i1, i2, labels, alphas, errorCache, G, bias, C, eps, tol, E2);
        if res == 1
            return
        end
    end
end
end

