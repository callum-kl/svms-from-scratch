function x0 = feasibleStart(y, C)
% Find a feasible start from the equality constraints
n = length(y);
pos = sum(y(y == 1));
frac = pos/(n - pos);
x0 = zeros(n,1);
for i = 1:n 
    if y(i) == 1
        x0(i) = C*(1-pos/n);
    else 
        x0(i) = C*(frac)*(1-pos/n);
    end
end
end

