function [ r_finalWeightMatrix, r_minERMS, r_optimalBasisFunctions ] = train_gd(p_lambda )

load global_variables.mat;

r_minERMS = 10000; % since it comes less than 1 for some value when I tested
r_optimalBasisFunctions = 2;

%for M = 3:15
M = 6;
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

previousERMS = 10000;
eta = 1;
previous_weightMatrix = zeros(M,1);
terminationCondition = false;
 
while terminationCondition == false
 
%For Minimized Error
weightMatrix = previous_weightMatrix + eta*(relevanceMatrixTraining(i,:)-(previous_weightMatrix*phiMatrix(i,:)))*transpose(phiMatrix(i,:));
weightMatrix = pinv(transpose(phiMatrix)*phiMatrix+p_lambda*eye(M))*transpose(phiMatrix)*relevanceMatrixTraining;
%Squared Error Function
phiWeight = (phiMatrix*weightMatrix - relevanceMatrixTraining);
squaredError = transpose(phiWeight)*phiWeight + 0.5*p_lambda*transpose(weightMatrix)*weightMatrix;
%Root Mean Square Error
eRMS = sqrt((2*squaredError)/totalTrainingRecords);
   
if eRMS > previousERMS
    eta = 0.5 * eta;
else if eRMS == previousERMS
    terminationCondition = true;    
end
 
previousERMS = eRMS;
previous_weightMatrix = weightMatrix;
 
end
 
end
    
if(eRMS < r_minERMS)
    r_minERMS = eRMS;
    r_optimalBasisFunctions = M;
    r_finalWeightMatrix = weightMatrix;
end

%end


end
