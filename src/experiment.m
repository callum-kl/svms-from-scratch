%% LOAD DATA
clear;
creditrating = readtable('CreditRating_Historical.dat');

data = table2array(creditrating(:,2:end-1)); labels = table2array(creditrating(:,end));

% Convert to ratings to numeric values
Y = zeros(length(labels), 1);
for i = 1:length(labels)
    rating = labels{i};
    switch rating
        case 'AAA'
            Y(i) = 1;
        case 'AA'
            Y(i) = 2;
        case 'A'
            Y(i) = 3;
        case 'BBB'
            Y(i) = 4;
        case 'BB'
            Y(i) = 5;
        case 'B'
            Y(i) = 6;
        case 'CCC'
            Y(i) = 7;
    end
end

% Pick out two classes to perform binary SVM
ind1 = 4; ind2 = 5;
indices = sort([find(Y==ind1); find(Y==ind2)]); n = length(indices);
X = zeros(n, 6); y = zeros(n,1); 
for i = 1:n
    index = indices(i);
    X(i,:) = data(index,:);
    if(Y(index) == ind1)
        y(i) = -1;
    elseif(Y(index) == ind2)
        y(i) = 1;
    end
end
X = X./(max(abs(X)));
numTrain = 400;
idx = randperm(n);
ytrain = y(idx(1:numTrain)); Xtrain = X(idx(1:numTrain),:);
ytest = y(idx(numTrain:end)); Xtest = X(idx(numTrain:end),:);

SAVE = false;

%% SMO Experiment 1 - Objective Function
eps = 1e-3; tol = 1e-3; maxIter = 500; verbose = 2;
[~, ~, info_smo1, ~] = SMO(Xtrain, ytrain, 10, 'poly', 2, eps, tol, maxIter, verbose, []);
[~, ~, info_smo2, ~] = SMO(Xtrain, ytrain, 100, 'gaussian', 0.5, eps, tol, maxIter, verbose, []);
%[~, ~, info_smo3, ~] = SMO(Xtrain, ytrain, 10, kType, 0.01, eps, tol, maxIter, verbose, []);
%%
figure(2); semilogy(abs(info_smo1.objEnergy - info_smo1.objEnergy(end)));
hold on; semilogy(abs(info_smo2.objEnergy - info_smo2.objEnergy(end)));
hold off; grid on;
legend('Polynomial', 'Gaussian');
title('SMO Convergence rate'); xlabel('Iteration'); ylabel('log|| g(\lambda^k) - g(\lambda^*) ||_2');

if SAVE
    saveas(2, './images/objectiveValuesC1_5.png');
end
%%
figure(2); semilogy(abs(info_smo2.objEnergy(2:end-1) - info_smo2.objEnergy(end))./abs(info_smo2.objEnergy(1:end-2) - info_smo2.objEnergy(end)))
grid on;
legend('Gaussian', 'Location', 'SouthEast');
title('SMO Convergence rate'); xlabel('Iteration'); ylabel('log|| g(\lambda^k) - g(\lambda^*) ||_2/|| g(\lambda^{k-1}) - g(\lambda^*) ||_2');

if SAVE
    saveas(2, './images/objectiveValuesC1_8.png');
end

%%
diffs1 = [];
for i = 1:length(info_smo1.xs)-1
    diffs1 = [diffs1 norm(info_smo1.xs{i} - info_smo1.xs{end})];
end

diffs2 = [];
for i = 1:length(info_smo2.xs)-1
    diffs2 = [diffs2 norm(info_smo2.xs{i} - info_smo2.xs{end})];
end
figure(2)
semilogy(diffs1)
hold on;
semilogy(diffs2)
grid on;
legend('Polynomial', 'Gaussian');
title('SMO Convergence rate'); xlabel('Iteration'); ylabel('log|| \lambda^k - \lambda^* ||_2');
if SAVE
    saveas(2, './images/objectiveValuesC1_6.png');
end
%%

figure(2); plot(abs(info_smo1.objEnergy(2:end-2) - info_smo1.objEnergy(end))./abs(info_smo1.objEnergy(1:end-3) - info_smo1.objEnergy(end)));
%hold on; plot(abs(info_smo2.objEnergy - info_smo2.objEnergy(end)));
%hold on; plot(abs(info_smo3.objEnergy - info_smo3.objEnergy(end)));
hold off; grid on;
legend('C = 0.1', 'C = 1', 'C = 10');
title('SMO Convergence rate'); xlabel('Iteration'); ylabel('log|| g(\lambda^k) - g(\lambda^*) ||_2');

if SAVE
    saveas(2, './images/objectiveValuesC1_1.png');
end

%% SMO Experiment 2 - Convergence rate

info_smoCur = info_smo1;
smoObjectiveDiff = zeros(length(info_smoCur.objEnergy)-1,1);
smoStar = info_smoCur.objEnergy(end);
for i = 1:(length(info_smoCur.objEnergy)-1)
    smoObjectiveDiff(i) = norm(info_smoCur.objEnergy(i) - smoStar);
end
figure(3); plot(log(smoObjectiveDiff)); grid on
title('Convergence rate Linear SMO'); xlabel('Iteration'); ylabel('|| g(?^k) - g(?^*) ||_2');

if SAVE
    saveas(3, './images/smoConvRateLinear.png');
end

%% LOG BARRIER EXPERIMENT 1 - varying mu
% Range of mu's
mus = [2,10,100]; dgaps = {}; desc = {};
% Constant params
C = 10; t = 1; maxIter = 120; kernelType = 'poly'; kernelParam = 2; verbose = 2;
for mu = mus
    [~, ~, info_logb, ~] = logBarrier(Xtrain, ytrain, C, mu, t, maxIter, kernelType, kernelParam, verbose, []);
    dgaps{end+1} = info_logb.dGap;
    desc{end+1} = compose('mu = %d', mu);
end
%%
figure(1)
SAVE = true;
for i = 1:length(dgaps)
    stairs((dgaps{i}), 'LineWidth', 0.8);
    hold on
end
set(gca, 'YScale', 'log'); grid on; legend('\mu = 2', '\mu = 10', '\mu = 100')
xlabel('Iteration'); ylabel('Feasibility gap m/t');
hold off;

if SAVE
    saveas(1, './images/logFeasibilityGap_1.png');
end


%% LOG BARRIER EXPERIMENT 2 - Newton Convergence

C = 20; mu = 10; t = 1; maxIter = 80; 
kernelType = 'gaussian'; kernelParam = 0.5; verbose = 2;

[ ~, ~, info_logb, ~] = logBarrier(Xtrain, ytrain, C, mu, t, maxIter, kernelType, kernelParam, verbose, []);

iterates = info_logb.ns; tVal = info_logb.t; logbDiff = [];
for i = 1:(length(iterates) - 1)
    logbDiff = [logbDiff norm(iterates(i) - iterates(end))];
end
%%
SAVE = false;
figure(5); plot(log(logbDiff)); grid on
title('Feasible Newton convergence'); xlabel('Iteration'); ylabel('log || g(x^k) - p^* ||_2');

if SAVE
    saveas(5, './images/NWconv_1.png');
end
%%
SAVE = true
figure(60)
plot(abs(info_logb.ns(2:end-1) - info_logb.ns(end))./abs(info_logb.ns(1:end-2) - info_logb.ns(end)))
grid on; xlabel('Iteration'); ylabel('|| g(x^k) - p^* ||_2/|| g(x^{k-1}) - p^* ||_2');
if SAVE
    saveas(60, './images/NWconv_2.png');
end

%% EXPERIMENT 3 - BARRIER OBJECTIVE FUNCTION CONVERGENCE
SAVE = false;
C = 20; mu = 5; t = 1; maxIter = 140; 
verbose = 2;
[ ~, ~, info_logb_1, ~] = logBarrier(Xtrain, ytrain, 100, mu, t, maxIter, 'gaussian', 0.5, verbose, []);
[ ~, ~, info_logb_2, ~] = logBarrier(Xtrain, ytrain, 10, mu, t, maxIter, 'poly', 2, verbose, []);
%[ ~, ~, info_logb_3, ~] = logBarrier(Xtrain, ytrain, C, mu, t, maxIter, 'linear', 1, verbose, []);
%%
SAVE = true;
figure(30)
semilogy(abs(info_logb_1.outer(1:end-2) - info_logb_1.outer(end)))
hold on;
semilogy(abs(info_logb_2.outer(1:end-2) - info_logb_2.outer(end)))
grid on; legend('Gaussian','Polynomial');
xlabel('Iteration K'); ylabel('log || g(\lambda^K) - g^* ||_2');
if SAVE
    saveas(30, './images/log_barrier_conv.png');
end
%%
SAVE = true;
figure(30)
plot(abs(info_logb_1.outer(2:end-2) - info_logb_1.outer(end))./abs(info_logb_1.outer(1:end-3) - info_logb_1.outer(end)))
hold on;
plot(abs(info_logb_2.outer(2:end-2) - info_logb_2.outer(end))./abs(info_logb_2.outer(1:end-3) - info_logb_2.outer(end)))
grid on; legend('Gaussian','Polynomial');
xlabel('Iteration K'); ylabel('|| g(\lambda^k) - g^* ||_2/|| g(\lambda^{k-1}) - g^* ||_2');
if SAVE
    saveas(30, './images/log_barrier_conv_1.png');
end
%%
semilogy(abs(info_logb_1.objEnergy(2:end-1) - info_logb_1.objEnergy(end))./abs(info_logb_1.objEnergy(1:end-2) - info_logb_1.objEnergy(end)))
hold on;

%% EXPERIMENT 4 - PERFORMANCE of Algorithms (CPU time)
eps = 1e-4; tol = 1e-4; maxIter = 2000; C = 5; mu = 20; t = 1;
kernelType = 'poly'; kernelParam = 1; verbose = 1;
[~, ~, info_smo, ~] = SMO(Xtrain, ytrain, C, kernelType, kernelParam, eps, tol, maxIter, verbose);
[ ~, ~, info_logb, ~] = logBarrier(Xtrain, ytrain, C, mu, t, maxIter, kernelType, kernelParam, verbose);

fprintf('SMO iterations: %d \n',info_smo.nIter);
fprintf('SMO time: %f \n',info_smo.Time);
fprintf('SMO time/iteration: %f \n',info_smo.Time/info_smo.nIter);
fprintf('LOGB iterations: %d \n',info_logb.nIter);
fprintf('LOGB time: %f \n',info_logb.Time);
fprintf('LOGB time/iterations: %f \n',info_logb.Time/info_logb.nIter);
fprintf('\n');

%% Experiment 4 - Value of C for Poly/Gauss
% Experiment to determine optimal value of C

cVals = 10.^(-2:0.5:3); accuracyPerP = {}; accuracyPerPSMO = {};
% Define fixed parameters.
mu = 20; t = 1; maxIter = 60; kernelType = 'poly';
eps = 1e-4; tol = 1e-4; maxIterSMO = 4000;
kernelParams = [1,2,3]; verbose = 1;


for p = kernelParams
    disp(p);
    accuracyPerC = [];
    accuracyPerCSMO = [];
    for C = cVals
        [a1, b1, info_logB, K1] = logBarrier(Xtrain, ytrain, C, mu, t, maxIter, kernelType, p, verbose);
        [a2, b2, info_smo, K2] = SMO(Xtrain, ytrain, C, kernelType, p, eps, tol, maxIterSMO, verbose);
        %ta1 = computeTestAccuracy(ytest, Xtest, ytrain, Xtrain, a1, b1, K1);
        %ta2 = computeTestAccuracy(ytest, Xtest, ytrain, Xtrain, a2, -b2, K2);
        %accuracyPerC = [accuracyPerC ta1];
        %accuracyPerCSMO = [accuracyPerCSMO ta2];
        accuracyPerC = [accuracyPerC info_logB.accValues];
        accuracyPerCSMO = [accuracyPerCSMO info_smo.accValues];
    end
    accuracyPerP{end+1} = accuracyPerC;
    accuracyPerPSMO{end+1} = accuracyPerCSMO;
end
%%
SAVE = true;
figure(7);
for i = 1:length(accuracyPerP)
    plot(log10(cVals), accuracyPerP{i});
    hold on;
end
grid on; legend('p = 1', 'p = 2', 'p = 3'); ylim([0.7,1.0]);
xlabel('C value'); ylabel('Train Accuracy'); title('Log Barrier training accuracy vs C value');
if SAVE
    saveas(7, './images/CpolyLOGB.png');
end
hold off

figure(8);
for i = 1:length(accuracyPerPSMO)
    plot(log10(cVals), accuracyPerPSMO{i});
    hold on;
end
grid on; legend('p = 1', 'p = 2', 'p = 3'); ylim([0.7,1.0]);
xlabel('C value'); ylabel('Train Accuracy'); title('SMO training accuracy vs C value')
if SAVE
    saveas(8, './images/CpolySMO.png');
end
hold off

%% Experiment 5 - objective function/ training/ test for Poly kernel p = 1,2,3
C = 10;
mu = 10; t = 1; maxIter = 80; 
eps = 1e-4; tol = 1e-4; maxIterSMO = 4000;
kernelParams = [1,2,3]; kernelType = 'poly'; verbose = 1;

LOGB_TRA = {}; LOGB_TEA = {}; LOGB_OBJ = {};
SMO_TRA = {}; SMO_TEA = {}; SMO_OBJ = {};


for p = kernelParams
    disp(p);
    [a1, b1, info_logB, K1] = logBarrier(Xtrain, ytrain, C, mu, t, maxIter, kernelType, p, verbose);
    [a2, b2, info_smo, K2] = SMO(Xtrain, ytrain, C, kernelType, p, eps, tol, maxIterSMO, verbose);
    
    ta1 = computeTestAccuracy(ytest, Xtest, ytrain, Xtrain, a1, b1, K1);
    ta2 = computeTestAccuracy(ytest, Xtest, ytrain, Xtrain, a2, -b2, K2);
    LOGB_TRA{end+1} = info_logB.accValues; LOGB_OBJ{end+1} = info_logB.objEnergy; LOGB_TEA{end+1} = ta1;
    SMO_TRA{end+1} = info_smo.accValues; SMO_OBJ{end+1} = info_smo.objEnergy; SMO_TEA{end+1} = ta2;
end

%%
disp(LOGB_TRA)
disp(LOGB_TEA)
disp(LOGB_OBJ)
disp(SMO_TRA)
disp(SMO_TEA)
disp(SMO_OBJ)


%% Experiment 6 - Value of C for Gauss
cVals = 10.^(-2:0.5:3); accuracyPerP = {}; accuracyPerPSMO = {};
% Define fixed parameters.
mu = 20; t = 1; maxIter = 60; kernelType = 'gaussian';
eps = 1e-4; tol = 1e-4; maxIterSMO = 2000;
kernelParams = [0.01,0.1,0.5,1]; verbose = 1;

for p = kernelParams
    disp(p);
    accuracyPerC = [];
    accuracyPerCSMO = [];
    for C = cVals
        [a1, b1, info_logB, K1] = logBarrier(Xtrain, ytrain, C, mu, t, maxIter, kernelType, p, verbose);
        [a2, b2, info_smo, K2] = SMO(Xtrain, ytrain, C, kernelType, p, eps, tol, maxIterSMO, verbose);
        ta1 = info_logB.accValues; % computeTestAccuracy(ytest, Xtest, ytrain, Xtrain, a1, b1, K1);
        ta2 = info_smo.accValues; %computeTestAccuracy(ytest, Xtest, ytrain, Xtrain, a2, -b2, K2);
        accuracyPerC = [accuracyPerC ta1];
        accuracyPerCSMO = [accuracyPerCSMO ta2];
    end
    accuracyPerP{end+1} = accuracyPerC;
    accuracyPerPSMO{end+1} = accuracyPerCSMO;
end
%%
SAVE = true;
figure(9);
for i = 1:length(accuracyPerP)
    plot(log10(cVals), accuracyPerP{i});
    hold on;
end
grid on; legend('\sigma = 0.01', '\sigma = 0.1', '\sigma = 0.5', '\sigma = 1',  'Location', 'SouthEast');
xlabel('C value (base 10)'); ylabel('Training Accuracy'); title('Log Barrier training accuracy vs C value')
if SAVE
    saveas(9, './images/CgaussLOGB.png');
end
hold off

figure(10);
for i = 1:length(accuracyPerPSMO)
    plot(log10(cVals), accuracyPerPSMO{i});
    hold on;
end
grid on; legend('\sigma = 0.01', '\sigma = 0.1', '\sigma = 0.5', '\sigma = 1', 'Location', 'SouthEast');
xlabel('C value (base 10)'); ylabel('Training Accuracy'); title('SMO training accuracy vs C value')
if SAVE
    saveas(10, './images/CgaussSMO.png');
end
hold off

%% Experiment 7 - objective function/ training/ test for Gauss kernel sigma = 0.01,0.5,1
C = 100;
mu = 10; t = 1; maxIter = 80; 
eps = 1e-4; tol = 1e-4; maxIterSMO = 1000;
kernelParams = [0.01,0.1,0.5,1]; kernelType = 'gaussian'; verbose = 1;

LOGB_TRA = {}; LOGB_TEA = {}; LOGB_OBJ = {};
SMO_TRA = {}; SMO_TEA = {}; SMO_OBJ = {};


for p = kernelParams
    disp(p);
    [a1, b1, info_logB, K1] = logBarrier(Xtrain, ytrain, C, mu, t, maxIter, kernelType, p, verbose);
    [a2, b2, info_smo, K2] = SMO(Xtrain, ytrain, C, kernelType, p, eps, tol, maxIterSMO, verbose);
    
    ta1 = computeTestAccuracy(ytest, Xtest, ytrain, Xtrain, a1, b1, K1);
    ta2 = computeTestAccuracy(ytest, Xtest, ytrain, Xtrain, a2, -b2, K2);
    LOGB_TRA{end+1} = info_logB.accValues; LOGB_OBJ{end+1} = info_logB.objEnergy; LOGB_TEA{end+1} = ta1;
    SMO_TRA{end+1} = info_smo.accValues; SMO_OBJ{end+1} = info_smo.objEnergy; SMO_TEA{end+1} = ta2;
end

%%
disp(LOGB_TRA)
disp(LOGB_TEA)
disp(LOGB_OBJ)
disp(SMO_TRA)
disp(SMO_TEA)
disp(SMO_OBJ)


%%
C = 50;
mu = 10; t = 1; maxIter = 30; 
eps = 1e-4; tol = 1e-4; maxIterSMO = 500;
kernelParams = [1,2,3]; kernelType = 'poly'; verbose = 1;
[a1, b1, info_logB, K1] = logBarrier(Xtrain, ytrain, C, mu, t, maxIter, kernelType, 1, verbose);
computeTestAccuracy(ytest, Xtest, ytrain, Xtrain, a1, b1, K1)

%% LB and SMO Experiment 2 - Objective plots for diff kernels?

verbose = 2; C = 1; kernelType = 'linear'; kernelParam = 0.01;
% Objective for SMO
eps = 1e-4; tol = 1e-4; para.deg = 1; maxIter = 600;
[alpha_smo, bias_smo, info_smo, K] = SMO(Xtrain, ytrain, C, kernelType, kernelParam, eps, tol, maxIter, verbose);
% Objective for Log Barrier
mu = 20; t = 1; maxIter = 80;
[alpha_logb, bias_logb, info_logb, K_logb] = logBarrier(Xtrain, ytrain, C, mu, t, maxIter, kernelType, kernelParam, verbose);
figure(2)
plot((info_smo.objEnergy));
hold on;
plot((info_logb.objEnergy));
hold off;

