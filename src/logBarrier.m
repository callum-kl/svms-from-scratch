function [x, bias, info, K] = logBarrier(X, y, C, mu, t, maxIter, kernelType, kernelParam, verbose, gram)
% Log barrier IP method to solve constrained dual SVM problem

% Write dual SVM problem as a quadratic program
% wish to solve:
%        min   0.5*x'Qx + p'x
%       subject to       Ax <= b        (inequality constraint)
%                       A_eq*x = b_eq
% In dual SVM case, Q is kernel matrix, inequalities are 0 <= x <= C
% and equalities are A_eq = y.' and b_eq = 0.

tStart = tic; 

K = @(x1, x2) Kernel(x1, x2, kernelType, kernelParam);
if isempty(gram)
    G = formGramMatrix(X, kernelType, kernelParam);
else 
    G = gram;
end

[Q, p, A, b] = quadratic_form(C, G, y);
A_eq = y.';


% Determine feasible starting point x0
x0 = feasibleStart(y, C);

% Define objective function by taking log of inequality constraints 
% and derivatives.
F.f = @(x, t) t*(0.5*x'*Q*x + p'*x) - sum(log(b - A*x));
F.df = @(x, t) t*(Q*x + p) + A'*(1./(b-A*x));
F.d2f = @(x, t) t*Q + A'*diag(1./(b-A*x).^2)*A; 

% Backtracking parameters
opts.c1 = 0.01;
opts.c2 = 0.8;
opts.maxLsIter = 10;
opts.eps = 1e-6;

% Complementary slackness tolerance
tol = 1e-6;
m = 2*length(y);
dGap = m/t;

nIter = 1;
x_k = x0;

info.objEnergy = [];
info.dGap = [];
info.accValues = [];
info.outer = [];

if verbose == 2
    info.ns = [];
    info.t = 0;
end

numChanged = 0;

while((dGap > tol) && (nIter < maxIter))

    % Take centering step
    [x_k, ~, lambda_k] = feasibleNewton(x_k, t, F, A, b, A_eq, opts);

    % Duality gap reduced
    if(lambda_k < opts.eps)
        dGap = m/t;
        t = mu*t;
        numChanged = numChanged + 1;
        info.outer = [info.outer objectiveFunction(y, x_k, G)];
    end

    if verbose == 2
        info.objEnergy = [info.objEnergy objectiveFunction(y, x_k, G)];
        info.dGap = [info.dGap dGap];
        bias = computeBias(x_k, y, G, C);
        info.accValues = [info.accValues computeAccuracy(y, x_k.', -bias, G)];
        if numChanged == 1
            info.ns = [info.ns F.f(x_k, t)];
            info.t = t;
        end
    end
    nIter = nIter + 1;
   
end

x = x_k;
tEnd = toc(tStart);
info.Time = tEnd;
info.nIter = nIter;

if verbose == 1       
    info.objEnergy = objectiveFunction(y, x_k, G);
    info.dGap = dGap;
    bias = computeBias(x_k, y, G, C);
    info.accValues = computeAccuracy(y, x_k.', -bias, G);
end

end

