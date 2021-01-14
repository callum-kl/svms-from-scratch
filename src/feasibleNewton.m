function [x, nu, lambda] = feasibleNewton(x, t, F, A, b, A_eq, opts)
% Takes a feasible newton step subject to equality constraints
% using backtracking line search.

grad = F.df(x,t);
hess = F.d2f(x,t);
% Form KKT system

KKT = [hess, A_eq.'; A_eq, 0];
RHS = [grad; 0];


% Invert to find solution
val = -KKT\RHS;

% Direction dx
dx = val(1:end-1);
% Equality constraint lagrange multiplier nu
nu = val(end);

lambda = -grad'*dx;

alpha = 1;
res = b - A*x;
diff = -A*dx;

% Backtracking to make sure inequality constraints are still satisfied
while min(res + alpha*diff) <= 0
    alpha = opts.c2*alpha;
end

% Standard Backtracking to find optimum
bIter = 0;
while (F.f(x + alpha*dx, t) - F.f(x, t) - opts.c1*alpha*lambda >= 0 && bIter < opts.maxLsIter)
    alpha = opts.c2*alpha;
    bIter = bIter + 1;
end

x = x + alpha*dx;
end

