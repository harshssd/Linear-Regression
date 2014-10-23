%   import data to variable data
data = importdata('dataset.txt');

[rows, columns] = size(data);

relevanceMatrix = data(:, 1);
featuresMatrix = data(:, 2:columns);
[totalRecords, numOfFeatures] = size(featuresMatrix);

relevanceMatrixTraining = data(1:rows-30000, 1);
featuresMatrixTraining = data(1:rows-30000, 2:columns);
[totalTrainingRecords, numOfFeaturesTraining] = size(featuresMatrixTraining);

relevanceMatrixValidation = data(rows-35000:rows-25000, 1);
featuresMatrixValidation = data(rows-35000:rows-25000, 2:columns);
[totalValidationRecords, numOfFeaturesValidation] = size(featuresMatrixValidation);

relevanceMatrixTest = data(rows-25000:rows, 1);
featuresMatrixTest = data(rows-25000:rows, 2:columns);
[totalTestRecords, numOfFeaturesTest] = size(featuresMatrixTest);

save global_variables.mat;

%----------------------------------------------------------------------
% Gaussian Basis Model

lambda = 0.3;
[weightMatrix_cfs, eRMS_cfs, M_cfs] = train_cfs(lambda);

[finalLambda_cfs, eRMS_cfs_validation] = validate_cfs(weightMatrix_cfs, M_cfs);

[ERMS_cfs_test] = test_cfs(finalLambda_cfs, M_cfs, weightMatrix_cfs);

%---------------------------------------------------------------------
% Stochastic Gradient Descent

[weightMatrix_gd, eRMS_gd_train, M_gd] = train_gd(finalLambda_cfs );

[finalLambda_gd, eRMS_gd_validation] = validate_gd(weightMatrix_gd, M_gd);

[ERMS_gd_test] = test_gd(finalLambda_cfs, M_cfs, weightMatrix_cfs);

%----------------------------------------------------------------------
% Final Output

fprintf('My ubit name is %s\n', 'hhassans');
fprintf('My student number is %d\n', 50098099);
fprintf('the model complexity M_cfs is %d\n', M_cfs);
fprintf('the model complexity M_gd is %d\n', M_gd);
fprintf('the regularization parameters lambda_cfs is %4.2f\n', finalLambda_cfs);
fprintf('the regularization parameters lambda_gd is %4.2f\n', finalLambda_gd);
fprintf('the root mean square error for the closed form solution is %4.2f\n', ERMS_cfs);
fprintf('the root mean square error for the gradient descent method is %4.2f\n', ERMS_gd_test);
% fprintf('the root mean square error for the Bayesian solution is %4.2f\n', rms_bs); %(if you do it)
