function y = convert_labels(label, y_train, num)
y = zeros(num, 1);
for i = 1:num
    if(y_train(i) == label)
        y(i) = 1;
    else
        y(i) = -1;
    end
end
end

