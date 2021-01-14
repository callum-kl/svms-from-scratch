function [res, alphas, errorCache, bias] = takeStep(i1, i2, labels, alphas, errorCache, G, bias, C, eps, tol, E2)
% Given the indices of two Lagrange multipliers, update their values.
res = 0;

if i1 == i2
    return
end

y1 = labels(i1);
alph1 = alphas(i1);
if alph1 > tol && alph1 < C - tol
    E1 = errorCache(i1);
else
    E1 = decisionFunction(i1, labels, alphas, bias, G) - y1;
end

y2 = labels(i2);
alph2 = alphas(i2);
    
s = y1*y2;

% WAY 1
if s > 0
    gamma = alph1 + alph2;
    if gamma > C
        L = gamma - C;
        H = C;
    else
        L = 0;
        H = gamma;
    end
else
    gamma = alph1 - alph2;
    if gamma > 0
        L = 0;
        H = C - gamma;
    else
        L = -gamma;
        H = C;
    end
end

if L == H
    res = 0;
    return
end

k11 = G(i1, i1);
k12 = G(i1, i2);
k22 = G(i2, i2);
eta = 2*k12 - k11 - k22;
if eta < 0
    a2 = alph2 + y2*(E2 - E1)/eta;
    if a2 < L
        a2 = L;
    elseif a2 > H
        a2 = H;
    end
else
    c1 = eta/2;
    c2 = y2*(E1 - E2) - eta*alph2;
    Lobj = c1*L^2 + c2*L;
    Hobj = c1*H^2 + c2*H;
    
    if Lobj < Hobj - eps
        a2 = L;
    elseif Lobj > Hobj + eps
        a2 = H;
    else
        a2 = alph2;
    end
end

if(a2 < 1e-8)
      a2 = 0;
elseif (a2 > C - 1e-8)
      a2 = C;
end

if abs(a2 - alph2) < eps*(a2 + alph2 + eps)
    res = 0;
    return
end

a1 = alph1 - s*(a2 - alph2);
%IS this needed?
if a1 < 0
   a2 = a2 + a1*s;
   a1 = 0;
elseif a1 > C
   a2 = a2 + s*(a1 - C);
   a1 = C;
end

b1 = bias + E1 + y1*(a1 - alph1)*k11 + y2*(a2 - alph2)*k12;
b2 = bias + E2 + y1*(a1 - alph1)*k12 + y2*(a2 - alph2)*k22;

if ((0 < a1)&&(a1 < C))
    bnew = b1;
elseif ((0 < a2)&&(a2 < C))
    bnew = b2;
else
    bnew = (b1 + b2)/2;
end

errorCache = errorCache + labels(i1)*(a1 - alph1).*G(i1,:) + ...
                labels(i2)*(a2 - alph2).*G(i2,:) - (bnew - bias);
errorCache(i1) = 0;
errorCache(i2) = 0;
alphas(i1) = a1;
alphas(i2) = a2;
bias = bnew;
res = 1;
end

