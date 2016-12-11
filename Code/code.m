
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%Comparing k-NN and Random Forest on Breast Cancer Wisconsin Dataset%%%
%%% Authors: Hannes Draxl and Ryan Nazareth, City, University of London%%%%
          
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Load the data, feature matrix and target variable

clear; clc;
load data % load the wisconsin breast cancer data set
df = data;  % define full data

% summary(data);  % print a summary table of the full data set

% define the columns as features 
X = table2array(df(:,3:32));
% define the target variable consisting of "benign" and "malignant"
y = table2array(df(:,2)); 
% display the class distribution 
countpercent = tabulate(y)  
% A slight class imbalance can be observed. (212 Malignant tumor samples
% vs. 357 benign)

% save the feature names
categories = data.Properties.VariableNames;
% slice of the first two columns containing the ID and target variable
categories = categories(3:32); 

%% Randomized Train-Test Split (70% training and 30% test)

num_points = size(X,1);
split_point = round(num_points*0.7);
rng(10);
% randomly order the num_points variable  
seq = randperm(num_points);
% split X and y into the same proportion of train and test set
X_train = X(seq(1:split_point),:);
X_test = X(seq(split_point+1:end),:);
y_train = y(seq(1:split_point));
y_test = y(seq(split_point+1:end));
fprintf('\nTraining data: %0.0f samples \nTest data: %0.0f samples',...
    size(X_train, 1), size(X_test, 1))

%% Feature Scaling 

% Apply feature scaling with mu=0 and stddev=1 to the features. 
% This is crucial for the subsequent k-NN algorithm as well as for PCA.

% Save the output as the scaled X_train matrix as well as the corresponding 
% mu and stddev of every feature column.

[X_train, mu, stddev] = scaler(X_train); 

% Standardize the test set with mu and std of the training set
% It is crucial to use mu and stddev from the training set as
% our test set represents an unseen set of samples.

for i=1:size(X_test, 2)
    X_test(:,i) = (X_test(:,i)-mu(1,i))/stddev(1,i);
end

fprintf('\nStandardizing training and test data with mu=0 and std=1:')
fprintf('\n')
fprintf('\nMean of standardized X_train: %.3f\nStd of standardized X_train: %.3f',...
    mean(mean(X_train(:,:))), mean(std(X_train(:,:))))
fprintf('\n')
fprintf('\nMean of standardized X_test: %.3f\nStd of standardized X_test: %.3f\n',...
    mean(mean(X_test(:,:))), mean(std(X_test(:,:))))

%% Machine Learning Algorithms

% 1) k-NN
% The first algorithm that is going to be applied to the data set is
% k-nearst neighbors. To find the optimal Hyperparameters (HP) for k-NN,
% bayesian HP tuning was utilised. The bayesian optimisation function
% displays two figures: The first one depicts the two HPs and the
% corresponding value of each iteration. The second graph shows the
% estimated and observed value of the Objective function (the
% miscallsification rate)

% Define the HPs and their parameter range for tuning:

% Number of neighbors k tuning parameter on a range of 1-100
neighbors = optimizableVariable('neighbors',[1,100],'Type','integer');
% Distance metric tuning parameter
distance = optimizableVariable('distance',{'correlation','cosine','hamming',...
    'jaccard','mahalanobis','cityblock','euclidean','minkowski','chebychev'...
    'seuclidean','spearman'},'Type','categorical');

% Set n_samples variable to the number of rows in the X_train matrix
n_samples = size(X_train, 1)
rng(10);
% Random partition for stratified 10-fold cross validation. Each fold
% roughly has the same class proportions.
cv = cvpartition(n_samples,'Kfold',10);

% Define the kfoldloss function. The input variables are the fitcknn function,
% the X_train matrix, the corresponding target values y_train, as well as the
% name-value pairs including the CVpartition and the two HPs. NSMethod is
% set to exhaustive.

fun = @(x)kfoldLoss(fitcknn(X_train, y_train, 'CVPartition', cv,...
                    'NumNeighbors', x.neighbors,...
                    'Distance', char(x.distance), 'NSMethod', 'exhaustive'));
                
% Set the Bayesian Hyperparameter search which takes the kfoldLoss function
% and the tuning variables as input.

results_knn = bayesopt(fun,[neighbors, distance],'Verbose',1,...
                   'MaxObjectiveEvaluations', 300)

%%

% save the best parameters from the previous HP search 
neighbors = results_knn.XAtMinObjective.neighbors;
distance = results_knn.XAtMinObjective.distance;

fprintf('\nThe best HPs found by bayesian optimisation are:')
fprintf('\nk=%.0f; distance metric=%s', neighbors, distance)
% The bayesian optimisation resulted in k=6 and distance=euclidean. These
% HPs achieved the lowest cross-validated misclassification error.

% Print the 10-fold CV Accuracy score
accuracy_knn = 100*(1 - results_knn.MinObjective);
fprintf('\nAccuracy score of KNN is: %0.3f', accuracy_knn)

%% 
% 2) Random Forests

% the same procedure as above, was applied to the Random forest algorithm.
% Here, we tuned three different HPs, the maximum number of randomly selected
% features at a given split node with a range between (1-10), the Split 
% Criterion as well as the number of Trees [betw. 10 and 200] in the 
% ensemble. We set the number of tuning rounds in the baysian object to 50 
% (Way higher settings did not improve the results).

max_features = optimizableVariable('max_features',[1,10],'Type','integer');
SplitCriterion = optimizableVariable('SplitCriterion',{'gdi','deviance'},...
            'Type','categorical');
NumLearningCycles = optimizableVariable('NumLearningCycles',[10,200],...
    'Type','integer');

% The rest of the code is exactly the same as for k-NN
n_samples = size(X_train, 1)
rng(10);
cv = cvpartition(n_samples,'Kfold',10);

fun = @(x)kfoldLoss(fitcensemble(X_train, y_train, 'CVPartition', cv,...
                    'Method', 'Bag'));
                
results_rf = bayesopt(fun,[max_features,SplitCriterion,NumLearningCycles],'Verbose',1,...
                   'MaxObjectiveEvaluations',50)
               
%%
               
max_features = results_rf.XAtMinObjective.max_features;
SplitCriterion = results_rf.XAtMinObjective.SplitCriterion;
NumLearningCycles = results_rf.XAtMinObjective.NumLearningCycles;

fprintf('\nThe best HPs found by bayesian optimisation are:')
fprintf('\nmax_features=%.0f; SplitCriterion=%s; NumLearningCycles=%.0f',...
    max_features, SplitCriterion, NumLearningCycles)
% The bayesian optimisation resulted in max_features=10, gini splitting 
% criterion and 157 trees in the ensemble. These
% HPs achieved the lowest cross-validated misclassification error.

% Print the 10-fold CV Accuracy score
accuracy_rf = 100*(1 - results_rf.MinObjective);
fprintf('\nAccuracy score of RF is: %0.3f', accuracy_rf)

%% Loss Plot
% Plot the 10-fold CV loss, the Out of bag (OOB)-loss and the test-loss at
% the different stages of the ensemble

% therefore we define a tree template with the optimal HPs
tree_opt = templateTree('NumVariablesToSample', max_features,...
                        'SplitCriterion', char(SplitCriterion));
 
rng(10);
% and refit the RF with this tree template as the base learner
rf_tuned = fitcensemble(X_train, y_train,...
                        'Method', 'Bag', 'NumLearningCycles',...
                        NumLearningCycles,'learners', tree_opt);
                    
% calculate the corssval score of the tuned ensemble
rf_tuned_cv = crossval(rf_tuned, 'KFold', 10);      

% and finally plot the three error curves in the sample figure
figure;
plot(loss(rf_tuned,X_test,y_test,'mode','cumulative'));
hold on;
plot(kfoldLoss(rf_tuned_cv,'mode','cumulative'),'r.');
plot(oobLoss(rf_tuned,'mode','cumulative'),'k--');
hold off;
xlabel('Number of trees');
ylabel('Classification error');
legend('Test','Cross-validation','Out of bag','Location','NE');

%% Plot Feature Importance

% This plot provides an estimation of the feature importance and could
% serve as a way of feature selection. It can be observed, that some
% features seem to be way more important compared to others. 

impOOB = oobPermutedPredictorImportance(rf_tuned);
figure;
bar(impOOB);
title('Unbiased Predictor Importance Estimates');
xlabel('Predictor variable');
ylabel('Importance');
h = gca;
h.XTickLabel = categories;
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';


%% PCA

% This section computes the principal components, loadings and % variance
% by applying the pca function on the training data. A pareto plot of the 
% first 10 components is produced to investigate how much variance in the 
% dataset is accounted for with dimensionality reduction. We have chosen to
% use 10 components since they account for 95% variance. 3d and 2d scatter 
% plots are produced for first 3 and 2 components respectively. We label 
% the pounts as benign or maligant in the 2d plot, we can visualise a clear
% separation between two cluster groups. The principal component scores are 
% the transformed training set after PCA. The principal component 
% coefficients are then used to transform the test data set.
% The output variables from the pca function are defined below:

% coeff = Principal component coefficients. Each column of coeff contains...
% ....   coefficients for one principal component
% latent = Principal component variances 
% score = Principal component scores
% explained = Percentage of total variance explained 
% space (rows = samples; Columns = components)
% tsquared = Hotelling's T-squared statistic for each observation in X
% mu = estimated mean of each variable in X

[coeff,score,latent,tsquared,explained,mu] = ...
    pca(X_train,'NumComponents',size(X_train,2),'Rows','all','Centered',false);

% Display the first 2 PCs

figure(1);
gscatter(score(:,1),score(:,2),y_train,'br','xo')
xlabel('1st Principal Component')
ylabel('2nd Principal Component')

% Plot cumulative explaned_variance 
labels = {'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9',...
    'PC10'};
figure(2)
pareto(explained)
xlabel('Principal Component')
ylabel('Variance Explained (%)')

% Projection of first two components on 2D plane 
figure(3)
biplot(coeff(:,1:2),'scores',score(:,1:2))%'varlabels',categories);
axis([-.26 0.6 -.51 .51]);

% 3D plot of first three components
figure(4)
biplot(coeff(:,1:3),'scores',score(:,1:3));
axis([-.26 0.8 -.51 .51 -.61 .81]);
view([30 40]);

% Reduce the feature set to variance_explained = 95 % 
%(in our case, the first 10 PCs)

sum(explained(1:10))
X_train_pca = score(:,1:10);
% the test data set must be reduced with the loadings from the training
% data set
score_test = X_test*coeff;
X_test_pca = score_test(:,1:10);

%% 1) KNN tuning with reduced X_train_pca

% Here we tune the k-NN algorithm using Bayes Optimisation on the 
% transformed training set after PCA. As previously, we tune across a
% range of neighbours and distance metrics. 

% define Hyperparameters and their parameter range for tuning
neighbors_pca = optimizableVariable('neighbors',[1,30],'Type','integer');
distance_pca = optimizableVariable('distance',{'correlation','cosine','hamming',...
    'jaccard','mahalanobis','cityblock','euclidean','minkowski','chebychev'...
    'seuclidean','spearman'},'Type','categorical');

n_samples = size(X_train_pca, 1)
rng(10);
cv = cvpartition(n_samples,'Kfold',10);

% Loss function that contains kfoldloss and fitcknn
fun = @(x)kfoldLoss(fitcknn(X_train_pca, y_train, 'CVPartition', cv,...
                    'NumNeighbors', x.neighbors,...
                    'Distance', char(x.distance), 'NSMethod', 'exhaustive'));
% Baysian Hyperparameter search
results_knn_pca = bayesopt(fun,[neighbors_pca, distance_pca],'Verbose',1,...
                   'AcquisitionFunctionName', 'expected-improvement-plus',...
                    'MaxObjectiveEvaluations',300)

%%

% save the best parameters from the previous HP search 
neighbors_pca = results_knn_pca.XAtMinObjective.neighbors;
distance_pca = results_knn_pca.XAtMinObjective.distance;

fprintf('\nThe best HPs found by bayesian optimisation are:')
fprintf('\nk=%.0f; distance metric=%s', neighbors_pca, distance_pca)
% The bayesian optimisation resulted in k=6 and distance=minkowski. These
% HPs achieved the lowest cross-validated misclassification error.

% Print the 10-fold CV Accuracy score
accuracy_knn_pca = 100*(1 - results_knn_pca.MinObjective);
fprintf('\nAccuracy score of KNN is: %0.3f', accuracy_knn_pca)

%%  2) RF tuning with reduced X_train_pca

% The same has been repeated for the RF algorithm with the X_train_pca 
% dataset.

max_features_pca = optimizableVariable('max_features',[1,10],'Type','integer');
SplitCriterion_pca = optimizableVariable('SplitCriterion',{'gdi','deviance'},...
            'Type','categorical');
NumLearningCycles_pca = optimizableVariable('NumLearningCycles',[10,200],...
    'Type','integer');

% The rest of the code is exactly the same as for k-NN
n_samples = size(X_train_pca, 1)
rng(10);
cv = cvpartition(n_samples,'Kfold',10);

fun = @(x)kfoldLoss(fitcensemble(X_train_pca, y_train, 'CVPartition', cv,...
                    'Method', 'Bag'));
                
results_rf_pca = bayesopt(fun,[max_features_pca,SplitCriterion_pca,NumLearningCycles_pca],'Verbose',1,...
                   'AcquisitionFunctionName','expected-improvement-plus',...
                   'MaxObjectiveEvaluations',50)
%%           
max_features_pca = results_rf_pca.XAtMinObjective.max_features;
SplitCriterion_pca = results_rf_pca.XAtMinObjective.SplitCriterion;
NumLearningCycles_pca = results_rf_pca.XAtMinObjective.NumLearningCycles;

fprintf('\nThe best HPs found by bayesian optimisation are:')
fprintf('\nmax_features=%.0f; SplitCriterion=%s; NumLearningCycles=%.0f',...
    max_features_pca, SplitCriterion_pca, NumLearningCycles_pca)

% Print the 10-fold CV Accuracy score
accuracy_rf_pca = 100*(1 - results_rf_pca.MinObjective);
fprintf('\nAccuracy score of RF is: %0.3f', accuracy_rf_pca)

%% Print model performances:
fprintf('\n')
fprintf('\n10 fold CV Accuracy score of KNN is : %0.3f%%', accuracy_knn) 
fprintf('\n10 fold CV Accuracy score of KNN (PCA)is : %0.3f%%', accuracy_knn_pca) 
fprintf('\n10 fold CV Accuracy score of RandomForest is : %0.3f%%', accuracy_rf) 
fprintf('\n10 fold CV Accuracy score of RandomForest (PCA) is : %0.3f%%', accuracy_rf_pca) 

%% KNN & RF on full training data + test data.

% Here we train k-NN and Random Forest models on the full training data 
% using thr optimised parameters computed from the previous sections. The 
% trained model is then applied to the unseen test data to compute a test 
% error and accruacy metric. 

% k-NN
rng(10);
knn = fitcknn(X_train, y_train, 'NumNeighbors', neighbors,...
              'distance', char(distance));
[y_pred_knn,score_knn] = predict(knn,X_test);  
test_error_knn = loss(knn,X_test,y_test);

% RF
rng(10);
tree_opt = templateTree('NumVariablesToSample', max_features,...
                        'SplitCriterion', char(SplitCriterion));
rf = fitcensemble(X_train, y_train,'Method','Bag',...
       'NumLearningCycles',NumLearningCycles,'learners', tree_opt); 
[y_pred_rf, score_rf] = predict(rf, X_test);  
test_error_rf = loss(rf,X_test,y_test);

%k-NN and Random Forest accuracy on test set
fprintf('\nk-NN test accuracy: %0.3f\n',(1 - test_error_knn) * 100)
fprintf('Random Forest test accuracy: %0.3f\n',(1 - test_error_rf) * 100)

%% KNN & RF on full training PCA data + test PCA data

% Here we train k-NN and Random Forest models on the full training data
% which has been reduced after PCA using the optimised parameters computed
% from tuning both models following PCA in the previous sections. 
% The trained model is then applied to the unseen test data to compute a 
% test error and accruacy metric. 

% k-NN
rng(10);
knn = fitcknn(X_train_pca, y_train, 'NumNeighbors', neighbors_pca,...
    'distance', char(distance_pca));
[y_pred_knn_pca, score_knn_pca] = predict(knn, X_test_pca);  
test_error_knn_pca = loss(knn,X_test_pca,y_test);

% RF
rng(10);
tree_opt = templateTree('NumVariablesToSample',max_features_pca,...
                        'SplitCriterion', char(SplitCriterion_pca));
rf = fitcensemble(X_train_pca, y_train,'Method', 'Bag',...
       'NumLearningCycles', NumLearningCycles_pca, 'learners', tree_opt); 
[y_pred_rf_pca,score_rf_pca] = predict(rf, X_test_pca);  
test_error_rf_pca = loss(rf,X_test_pca,y_test);

%k-NN and Random Forest accuracy on test set
fprintf('\nk-NN PCA test accuracy: %0.3f\n',(1 - test_error_knn_pca) * 100)
fprintf('Random Forest PCA test accuracy: %0.3f\n',(1 - test_error_rf_pca) * 100)

%% Confusion Matrix, Precision, Recall, F1 Score for KNN and RF

% Additional metrics are computed for evaluating k-NN and Random Forest 
% performance on the test set. Given that our dataset is slightly
% imbalanced, these metrics will further support our conclusions rather 
% than relying on the accuracy metric alone.

% Confusion Matrix: Each column of the matrix represents the instances in a
% predicted class while each row represents the instances in an actual
% class. In this case we have a 2x2 matrix where the first column is the 
% predicted poisitive condition (malignant) and second column is the predicted 
% predicted negative condition (benign).

% Precision, Recall and F1 score: Precision is the ability of the 
% classifier not to label a positive sample as negative (benign). Recall is 
% the ability of the classifier to find all the positive samples (malignant 
% classes). 
% The F1 score is the weighted harmonic mean of the precision and recall.
% (F1 score = 2x(precision x recall)/(precision + recall))
% The higher the F1 score (maximum of 1), the better the model performance. 


[Cknn, order] = confusionmat(y_test, y_pred_knn);
[Crf, order] = confusionmat(y_test, y_pred_rf);

precision_knn = Cknn(2,2)./(Cknn(2,2)+Cknn(1,2));
recall_knn =  Cknn(2,2)./(Cknn(2,1)+Cknn(2,2));
f1Score_knn =  2*(precision_knn.*recall_knn)./(precision_knn+recall_knn);

precision_rf =Crf(2,2)./(Crf(2,2)+Crf(1,2));
recall_rf =  Crf(2,2)./(Crf(2,1)+Crf(2,2));
f1Score_rf =  2*(precision_rf.*recall_rf)./(precision_rf+recall_rf);

fprintf('\nKNN Test Set Performance without PCA:')
fprintf('\nPrecision: %0.3f\n', precision_knn) 
fprintf('Recall: %0.3f\n', recall_knn) 
fprintf('F1: %0.3f\n', f1Score_knn) 
fprintf('\nRF Test Set Performance without PCA:')
fprintf('\nPrecision: %0.3f\n', precision_rf) 
fprintf('Recall: %0.3f\n', recall_rf) 
fprintf('F1: %0.3f\n', f1Score_rf)

%% Confusion Matrix, Precision, Recall, F1 Score for KNN and RF (PCA)

% Same process as described above is applied for computing metrics for k-NN
% and RF model performance on the PCA reduuced test set  

[Cknn_pca, order] = confusionmat(y_test, y_pred_knn_pca);
[Crf_pca, order] = confusionmat(y_test, y_pred_rf_pca);

precision_knn_pca =Cknn_pca(2,2)./(Cknn_pca(2,2)+Cknn_pca(1,2));
recall_knn_pca =  Cknn_pca(2,2)./(Cknn_pca(2,1)+Cknn_pca(2,2));
f1Score_knn_pca =  2*(precision_knn_pca.*recall_knn_pca)./(precision_knn_pca+recall_knn_pca);

precision_rf_pca = Crf_pca(2,2)./(Crf_pca(2,2)+Crf_pca(1,2));
recall_rf_pca =  Crf_pca(2,2)./(Crf_pca(2,1)+Crf_pca(2,2));
f1Score_rf_pca =  2*(precision_rf_pca.*recall_rf_pca)./(precision_rf_pca+recall_rf_pca);

fprintf('\nKNN Test Set Performance with PCA:')
fprintf('\nPrecision: %0.3f', precision_knn_pca) 
fprintf('\nRecall: %0.3f', recall_knn_pca) 
fprintf('\nF1: %0.3f\n', f1Score_knn_pca) 
fprintf('\nRF Test Set Performance with PCA:')
fprintf('\nPrecision: %0.3f', precision_rf_pca) 
fprintf('\nRecall: %0.3f', recall_rf_pca) 
fprintf('\nF1: %0.3f\n', f1Score_rf_pca)
