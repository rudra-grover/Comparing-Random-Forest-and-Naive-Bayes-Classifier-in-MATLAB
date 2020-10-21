%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%               README:
% - This MATLAB Code file does both training and testing of Naive Bayes 
% and Random Forest models on the modified Wine Quality dataset from 
% the UCI ML Repository.
% - This dataset has 3000 rows and 13 columns
% - Minor preprocessing was carried out in python to create the dataset
% - wines.csv : Dataset
% - Script splits wines.csv into train and test
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clear
close all

%% Import data
wines = readtable("wines.csv");

wines(:,1:end-1) = varfun(@normalize, wines, 'InputVariables', 1:width(wines)-1); %normalise everything apart from response variable

%% Define attributes
predictorNames = {'fixed_acidity' 'volatile_acidity' 'citric_acid' 'residual_sugar' 'chlorides'...
    'free_sulfur_dioxide' 'total_sulfur_dioxide' 'density' 'ph' 'sulphates' 'alcohol'};

classNames = unique(wines.quality)'; % Class order

for p = 1:size(classNames,2)
     classCounts{1,p} = sum(wines.quality(:) == classNames(p));
end

%% Identify and remove classes with less than 10 instances (to ensure 10-fold split) 
for p = 1:size(classNames,2)
    if classCounts{1,p} < 10
        wines(wines.quality == classNames(1,p), :) = [];
        classNames(:,p) = [];
        classCounts(:,p) = [];
    end
end
       
%% Partition data
% partition data into 80-20 train and test
cvpt = cvpartition(wines.quality,'Holdout',0.2)  
trainIdx = training(cvpt); 
testIdx = test(cvpt); 

trainData = wines(trainIdx,:);
Xtrain = table2array(trainData(:,1:end-1));
Ytrain = table2array(trainData(:,end));
testData = wines(testIdx,:);
Xtest = table2array(testData(:,1:end-1));
Ytest = table2array(testData(:,end));

%% RFC changes

Y_p = Ytrain';                      % Transpose Ytrain to correct format for classifier
addpath(genpath('libs\'));          % Addpath for required classes

%% NB prior
inputPrior = cell2mat(classCounts) ./ sum(cell2mat(classCounts)); 

%% Naive Bayes PCA
numData = table2array(wines(:,1:end-1));
resp = wines.quality;
[pcs,scrs,~,~,pexp] = pca(numData); % perform pca on predictors

% sampling 9 princial components based on the pareto chart
trainData = scrs(trainIdx,1:9);
trainDataResp = resp(trainIdx,:);
testData = scrs(testIdx,1:9);
testDataResp = resp(testIdx,:);

% pareto chart for pca
figure(1)
pareto(pexp)
saveas(figure(1),'Comparison_Figure1.jpg')
%% Fitting NB model + Train/Test Error
mdl_fit = fitcnb(trainData,trainDataResp,'DistributionNames','kernel','Kernel','box','ClassNames',classNames,'Prior',inputPrior);
mdl_trainerr = resubLoss(mdl_fit); %Train classification error
mdl_testerr = loss(mdl_fit,testData,testDataResp); %Test classification error
[mdl_predictedQuality, score] = predict(mdl_fit,testData); 
[cm, grp] = confusionmat(testDataResp,mdl_predictedQuality,'Order',classNames);

%% Fitting RF model
% Training Accuracy
X_n = Xtrain'; 
myForest = RForestClass(200); % Set number of trees to 200
tic % start timer for training time
forest_1 = myForest.trainRF(Xtrain', Y_p); % build forest
RFC_time_train = toc % end timer for training time
outD = []; % scores
pL = []; % predictions 
for i = 1:size(Y_p,2)
    [pLabel, outNode]  = myForest.predictRF(X_n(:, i));
    outD = [outD; outNode.m_hist(2,:)];
    pL = [pL pLabel];
end
err_train = immse(Y_p, pL); % training error (Mean Squared Error)

% Testing Accuracy
% set min_samples split to 5 and max depth to 25 
% increasing the max depth gives a memory requirements error
myTree = RTreeClass(5, 25); 
tic % start timer 1 for test time
myTree.trainTree(X_n, Y_p);
RFC_time_test1 = toc % end timer 1 for test time
outC = [];
X_test = Xtest'
Y_test = Ytest'
for i = 1:size(Y_test,2)
    outNode  = myTree.predictTree(X_test(:, i));
    outC = [outC; outNode.m_hist(2,:)];
end

myForest = RForestClass(200); % num trees is 200
tic % start timer 2 for test time
myForest.trainRF(X_n, Y_p); 
RFC_time_test2 = toc % end timer 2 for test time
outD = []; % scores
pL = []; % predictions
for i = 1:size(Y_test,2)
    [pLabel, outNode]  = myForest.predictRF(X_test(:, i));
    outD = [outD; outNode.m_hist(2,:)];
    pL = [pL pLabel];
end

%% RF Compute Metrics - Classification Error, Accuracy, Precision, Recall, F1 Score
CF = confusionmat(Y_test, pL); % Compute confusion matrix
err = immse(Y_test, pL); % Compute classification error
err_train

CF(isnan(CF))=0; % Convert NaNs in confusion matrix to 0

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

% Convert scores from cell to matrices for matrix operations
Recall = cell2mat(Recall);
Precision = cell2mat(Precision);
F1Score = cell2mat(F1Score);
C1 = cell2mat(C1);

% Compute weighted precision, recall and F1 score
multiRecall = Recall*C1; % total recall
multiPrecision = Precision*C1; % total precision
multiF1 = F1Score*C1; % total F1 score
RFC_weightedPrecision = trace(multiPrecision)/600 % average weighted Recall
RFC_weightedRecall = trace(multiRecall)/600 % average weighted precision
RFC_weightedF1Score = trace(multiF1)/600 % average weighted F1 score

%Accuracy scores (arithmetic mean)
RFC_Accuracy = sum(diag(CF)) / sum([CF(:)])

% Training and testing time for model
RFC_time_train
RFC_time_test = RFC_time_test1 + RFC_time_test2


%% NB Compute Other Metrics 
              
for p = 1:size(classNames,2)
    loopPerf{p,1} = cm(p,p) / sum([cm(p,:)]); %recall loop
    loopPerf{p,2} = cm(p,p) / sum([cm(:,p)]); %precision loop
    loopPerf{p,3} = (2 * loopPerf{p,1} * loopPerf{p,2}) ./ (loopPerf{p,1} + loopPerf{p,2});   %F1-score loop        
end
perfMeasures = loopPerf;  %Store measures: columns = Recall, Precision and F1-score per prior cell; rows = Classes
perfMeasures = cell2mat(perfMeasures); %convert to matrix 
perfMeasures(isnan(perfMeasures))=0; %convert NaN to matrix

%Multiply performance measures with class weights
classCountsMat = cell2mat(classCounts);
totRecall = perfMeasures(:,1) .* classCountsMat'; %total precision
totPrecision = perfMeasures(:,2) .* classCountsMat'; %total precision
totF1Score = perfMeasures(:,3) .* classCountsMat'; %total F1  

%Sum total measure and divide by total class number to get weighted score
NB_weightedRecall = sum([totRecall(:,1)]) ./ sum(classCountsMat) %weighted average recall     
NB_weightedPrecision = sum([totPrecision(:,1)]) ./ sum(classCountsMat) %weighted average precision
NB_weightedF1Score = sum([totF1Score(:,1)]) ./ sum(classCountsMat) %weighted average F1 score

%Calculate Accuracy
NB_Accuracy = sum(diag(cm)) / sum([cm(:)]) %accuracy

% Training and testing time for model
NBtrainHandle = @() fitcnb(trainData,trainDataResp,'DistributionNames','kernel','Kernel','box','ClassNames',classNames,'Prior',inputPrior);
NB_trainTime = timeit(NBtrainHandle)

NBtestHandle = @() predict(mdl_fit,testData);
NB_testTime = timeit(NBtestHandle)

%Training and testing errors
NB_classification_trainerr = mdl_trainerr  %Train classification error
NB_classification_testerr = mdl_testerr  %Test classification error

[train_predictedQuality, trainscore] = predict(mdl_fit,trainData); % Used only to get mean square error at training

NB_MSE_train = immse(trainDataResp,train_predictedQuality) %Train mean squared error
NB_MSE_test = immse(testDataResp,mdl_predictedQuality) %Test mean squared error

%% NB ROC Curve
%Create an index loop to ignore chosen class column and find the maximum of others
tloop = p + (p-2); %to find maximum loops required

for t = 0:tloop
    modulo(t+1,p) = mod(t,p) + 1;
end

ndimsp1 = ndims(score)+1;   

%Calculate diffscore and ROC components
for tt = 1:p
    diffscore{tt} = score(:,tt) - max(cat(ndimsp1,score(:,modulo(tt+1,p)),score(:,modulo(tt+2,p)),...
                                                 score(:,modulo(tt+3,p)),score(:,modulo(tt+4,p)),score(:,modulo(tt+5,p))),[],ndimsp1);
    [Xnb{tt},Ynb{tt},Tnb{tt},AUCnb{tt}] = perfcurve(testDataResp,diffscore{tt},classNames(p));
                                            
end

%Plot 
figure(2)
hold on
for tt = 1:p
    plot(Xnb{tt},Ynb{tt});
end
title('Naive Bayes: Multiclass ROC Curves - One vs All Approach')
xlabel('False Positive Rate')
ylabel('True Positive Rate')
x = [0,1];, y = [0,1];
pl = line(x,y);
pl.LineStyle = '--';
pl.Color = 'blue';
legend('rating 3', 'rating 4', 'rating 5', 'rating 6', 'rating 7', 'rating 8', 'Location', 'best') 
hold off
saveas(figure(2),'Comparison_Figure2.jpg')

%% ROC RFC
ndimsrfc = ndims(outD)+1;   

%Calculate diffscore and ROC components
for tt = 1:p
    diffscore{tt} = outD(:,tt) - max(cat(ndimsrfc,outD(:,modulo(tt+1,p)),outD(:,modulo(tt+2,p)),...
                                                 outD(:,modulo(tt+3,p)),outD(:,modulo(tt+4,p)),outD(:,modulo(tt+5,p))),[],ndimsrfc);
    [Xrfc{tt},Yrfc{tt},Trfc{tt},AUCrfc{tt}] = perfcurve(Y_test,diffscore{tt},classNames(p));                                          
end

%Plot 
figure(3)
hold on
for tt = 1:p
    plot(Xrfc{tt},Yrfc{tt});
end
title('Random Forest: Multiclass ROC Curves - One vs All Approach')
xlabel('False Positive Rate')
ylabel('True Positive Rate')
x = [0,1];, y = [0,1];
pl = line(x,y);
pl.LineStyle = '--';
pl.Color = 'blue';
legend('rating 3', 'rating 4', 'rating 5', 'rating 6', 'rating 7', 'rating 8', 'Location', 'best') 
hold off
saveas(figure(3),'Comparison_Figure3.jpg')

%% Average AUC
avgAUCnb = mean(cell2mat(AUCnb))
avgAUCrfc = mean(cell2mat(AUCrfc))

