function [Q,p,A,b] = quadratic_form(C,K,y)

[n,~] = size(K);
Q =  diag(y)*K*diag(y);
p =  -ones(n,1);
b =  [C*ones(n,1); zeros(n,1)];
A =  [eye(n);-eye(n)];

end

