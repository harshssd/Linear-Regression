function [ r_finalWeightMatrix, r_minERMS, r_optimalBasisFunctions ] = train_cfs(p_lambda)

load global_variables.mat;

minERMS = 10000; % since it comes less than 1 for some value when I tested
optimalBasisFunctions = 2;

%for M = 3:15
M = 4;
chunkSize = ceil(totalTrainingRecords/(M-1));

mu = zeros(M, numOfFeaturesTraining);

for i=2:M
  if((i*chunkSize)<totalTrainingRecords)
     mu(i,:) = mean(featuresMatrixTraining((i-2)*chunkSize+1:(i-1)*chunkSize,:))*(i*1.3); 
  else
     mu(i,:) = mean(featuresMatrixTraining((i-2)*chunkSize+1:totalTrainingRecords, :))*(i*1.3);
  end
end

phiMatrix = zeros(totalTrainingRecords, M);
for i=1:totalTrainingRecords
        phiMatrix(i,1) = 1;
end
for j=2:M
    for i=1:totalTrainingRecords
        xMinusMu = (featuresMatrixTraining(i,:)-mu(j,:));
        sigmaSquare = var(featuresMatrixTraining(i,:))*eye(numOfFeaturesTraining);
        phiMatrix(i,j) = exp(-(xMinusMu*pinv(sigmaSquare)*transpose(xMinusMu)));
    end
end

%For Minimized Error
weightMatrix = pinv(transpose(phiMatrix)*phiMatrix+p_lambda*eye(M))*transpose(phiMatrix)*relevanceMatrixTraining;

%Squared Error Function
phiWeight = (phiMatrix*weightMatrix - relevanceMatrixTraining);
    
squaredError = transpose(phiWeight)*phiWeight + 0.5*p_lambda*transpose(weightMatrix)*weightMatrix;

%Root Mean Square Error
eRMS = sqrt((2*squaredError)/totalTrainingRecords);
    
if(eRMS < minERMS)
    r_minERMS = eRMS;
    r_optimalBasisFunctions = M;
    r_finalWeightMatrix = weightMatrix;
end

%end

end

