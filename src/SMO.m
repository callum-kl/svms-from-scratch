function [alphas, bias, info, K] = SMO(points, labels, C, kType, kParam, eps, tol, maxIter, verbose, gram)
% SMO - Sequential Minimal Optimisation
% Reference: This implementation is heavily based upon the
% pseudocode in Platt (1998).
% 
% INPUT
% points: input points (training)
% labels: classification labels
% C : upper bound on langrange multipliers
% struct containing convergence criterion, kernel type and 
% kType: string specifying kernel type, e.g
%                   'linear', gaussian', 'poly'
% para: struct containing optional parameters
% eps: convergence criterion value
% tol : tolerance value for multipliers 
%
% OUTPUT
% alphas: langrange multipliers
% info: structure containing convergence history

tStart = tic;

K = @(x1, x2) Kernel(x1, x2, kType, kParam);

if isempty(gram)
    G = formGramMatrix(points, kType, kParam);
else
    G = gram;
end

N = length(labels);
% Initialise Lagrange multipliers
alphas = zeros(1, N);
% All alphas start as zero so Ei = -yi
errorCache = zeros(1, N);
bias = 0;

examineAll = 1;
numChanged = 0;
iter = 1;

info.objEnergy = zeros(1, maxIter);
info.accValues = zeros(1, maxIter);
info.xs = {};
info.nIter = 0;


while ((examineAll || numChanged > 0) && (iter < maxIter))
    % Compute value of objective function and store
    curEnergy = objectiveFunction(labels, alphas, G);
    if (verbose == 2)
        info.objEnergy(:,iter) = curEnergy;
        info.xs{end+1} = alphas;
        acc = computeAccuracy(labels, alphas, bias, G);
        info.accValues(:,iter) = acc;
    end

    numChanged = 0;
    if examineAll
        for i = 1:N
            [res, alphas, errorCache, bias] = examineExample(i, labels, alphas, errorCache, C, bias, eps, tol, G);
            numChanged = numChanged + res;
        end
    else 
        % alphas or errors?
        nonBoundIndices = find(alphas > tol & alphas < C - tol);
        for i = nonBoundIndices
            [res, alphas, errorCache, bias] = examineExample(i, labels, alphas, errorCache, C, bias, eps, tol, G);
            numChanged = numChanged + res;
        end
    end
    
    if examineAll == 1 
        examineAll = 0;
    elseif numChanged == 0
        examineAll = 1;
    end
    iter = iter + 1;
end

if (verbose == 2) 
    info.accValues = info.accValues(1:iter-1);
    info.objEnergy = info.objEnergy(1:iter-1);
else
    info.objEnergy = objectiveFunction(labels, alphas, G);
    info.accValues = computeAccuracy(labels, alphas, bias, G);
end
tEnd = toc(tStart);
info.Time = tEnd;
info.nIter = iter;
end

