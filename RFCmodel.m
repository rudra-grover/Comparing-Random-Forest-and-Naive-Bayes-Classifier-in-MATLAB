close all
clear all
clc

% reading in the dataset
wine = readtable('C:\Users\Rudra\all_wines_quality.csv');

X = table2array(wine(:,1:12));      % Separating predictor variables
Y = wine.quality;                   % Separating the targer vector

%% Define attributes
predictorNames = {'fixed_acidity' 'volatile_acidity' 'citric_acid' 'residual_sugar' 'chlorides'...
    'free_sulfur_dioxide' 'total_sulfur_dioxide' 'density' 'ph' 'sulphates' 'alcohol'};

classNames = unique(wine.quality)'; % Class order
classCounts = cell(1,size(classNames,2));

for p = 1:size(classNames,2)
     classCounts{1,p} = sum(wine.quality(:) == classNames(p));
end

%% Create Train and Test data
X_n = normalize(X);                 % Normalising predictor variables
cv = cvpartition(Y, 'holdout', .2); % Splitting dataset into training and testing - 80:20
%cv = cvpartition(Y, 'Kfold', 10);
Xtrain = X_n(cv.training,:);        % Create matrix of X values for training data
Ytrain = Y(cv.training,1);          % Create matrix of Y values for training data
Xtest = X_n(cv.test,:);             % Create matrix of X values for testing data
Ytest = Y(cv.test,1);               % Create matrix of X values for testing data

% Compute PCA for training predictors
[coeff,scoreTrain,~,~,explained,mu] = pca(Xtrain); 

% Plot pareto chart for PCA
figure
pareto(explained)
title('Pareto Chart for Principal Components')

sum_explained = 0;
idx = 0;
% Compute number of significant Principal Components
while sum_explained < 95
    idx = idx + 1;
    sum_explained = sum_explained + explained(idx);
end
% idx
scoreTrain9 = scoreTrain(:,1:idx);  % Create PC matrix
X_n = scoreTrain9';                 % Transpose matrix to correct format for classifier
Y_p = Ytrain';                      % Transpose Ytrain to correct format for classifier
addpath(genpath('libs\'));          % Addpath for required classes

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% TRAINING ACCURACY

myForest = RForestClass(200);
tic
forest_1 = myForest.trainRF(scoreTrain9', Y_p);
time_train = toc
outD = [];
pL = [];
for i = 1:size(Y_p,2)
    [pLabel, outNode]  = myForest.predictRF(X_n(:, i));
    outD = [outD; outNode.m_hist(2,:)];
    pL = [pL pLabel];
end

figure('name','Training Data');
scatter3(X_n(1,:), X_n(2,:), Y_p, 36, [0 0 1]);
hold on;
scatter3(X_n(1,:), X_n(2,:), pL, 18, [1 0 0]);
title('Training data');
xlabel('X Attribute 1'); % x-axis label
ylabel('X Attribute 2'); % y-axis label
zlabel('Y'); % z-axis label
legend('Labels', 'Training Classification');
hold off;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% TESTING ACCURACY

% PCA transform the Test dataset
scoreTest9 = (Xtest-mu)*coeff(:,1:idx);

myTree = RTreeClass(5, 25);
tic
myTree.trainTree(scoreTrain9', Y_p);
time_test1 = toc
scTest = scoreTest9';
outC = [];
%X_test = Xtest'
Y_test = Ytest'
for i = 1:size(Y_test,2)
    outNode  = myTree.predictTree(scTest(:, i));
    outC = [outC; outNode.m_hist(2,:)];
end

myForest = RForestClass(200);
tic
myForest.trainRF(scoreTrain9', Y_p);
time_test2 = toc
outD = [];
pL = [];
for i = 1:size(Y_test,2)
    [pLabel, outNode]  = myForest.predictRF(scTest(:, i));
    outD = [outD; outNode.m_hist(2,:)];
    pL = [pL pLabel];
end

figure('name','Testing Data');
scatter3(scTest(1,:), scTest(2,:), Y_test, 36, [0 0 1]);
hold on;
scatter3(scTest(1,:), scTest(2,:), pL, 18, [1 0 0]);
title('Testing data');
xlabel('X Attribute 1'); % x-axis label
ylabel('X Attribute 2'); % y-axis label
zlabel('Y'); % z-axis label
legend('Labels', 'Testing Classification');

%% Compute Metrics - Classification Error, Accuracy, Precision, Recall, F1 Score
CF = confusionmat(Y_test, pL); % Compute confusion matrix
err = immse(Y_test, pL); % Compute classification error
    
CF(isnan(CF))=0; % Convert NaNs in comfusion matrix to 0

for r = 1:size(CF,2)
    Precision{r,1} = CF(r,r) / sum([CF(r,:)]); %recall loop
    Recall{r,1} = CF(r,r) / sum([CF(:,r)]); %precision loop
    F1Score{r,1} = (2 * Recall{r,1} * Precision{r,1}) ./ (Recall{r,1} + Precision{r,1});   %F1-score loop
    C1{1,r} = sum([CF(:,r)]); % Sums of columns stored in row 1 of C1
end

for rowtoDelete = 2:size(C1)
    C1(rowtoDelete, :) = []; % Removing rows with no values (all but row 1)
end
% Convert NaN to zeros
F1Score = cellfun(@(M) subsasgn(M, substruct('()', {isnan(M)}), 0), F1Score, 'uniform', 0);
Precision = cellfun(@(M) subsasgn(M, substruct('()', {isnan(M)}), 0), Precision, 'uniform', 0);
Recall = cellfun(@(M) subsasgn(M, substruct('()', {isnan(M)}), 0), Recall, 'uniform', 0);

% Convert scores from cell to matrices
Recall = cell2mat(Recall);
Precision = cell2mat(Precision);
F1Score = cell2mat(F1Score);
C1 = cell2mat(C1);

% Compute weighted precision, recall and F1 score
multiRecall = Recall*C1;
multiPrecision = Precision*C1;
multiF1 = F1Score*C1;
weightedPrecision = trace(multiPrecision)/600
weightedRecall = trace(multiRecall)/600
weightedF1Score = trace(multiF1)/600

%Accuracy scores (arithmetic mean)
Accuracy = sum(diag(CF)) / sum([CF(:)])

% Training and testing time for model
time_train
time_test = time_test1 + time_test2

%% Plot ROC Curves

tloop = p + (p-2); %to find maximum loops required

for t = 0:tloop
    modulo(t+1,p) = mod(t,p) + 1;
end

ndimsrfc = ndims(outD)+1;   

%Calculate diffscore and ROC components
for tt = 1:p
    diffscore{tt} = outD(:,tt) - max(cat(ndimsrfc,outD(:,modulo(tt+1,p)),outD(:,modulo(tt+2,p)),...
                                                 outD(:,modulo(tt+3,p)),outD(:,modulo(tt+4,p)),outD(:,modulo(tt+5,p))),[],ndimsrfc);
    [Xrfc{tt},Yrfc{tt},Trfc{tt},AUCrfc{tt}] = perfcurve(Y_test,diffscore{tt},classNames(p));                                          
end

%Plot 
figure(4)
hold on
for tt = 1:p
    plot(Xrfc{tt},Yrfc{tt});
end
title('Multiclass ROC Curves - One vs All Approach')
xlabel('False Positive Rate')
ylabel('True Positive Rate')
legend('rating 3', 'rating 4', 'rating 5', 'rating 6', 'rating 7', 'rating 8') 
hold off
