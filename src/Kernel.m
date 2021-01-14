function k = Kernel(x1, x2, type, param)

switch type
    case 'linear'
        k = dot(x1,x2);
    case 'gaussian'
        k = exp(-(1/param)*norm(x1-x2)^2);
    case 'poly'
        k = dot(x1, x2)^param;
    otherwise
        warning('Invalid kernel type specified')
end

