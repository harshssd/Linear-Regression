function [ r_eRMS ] = test_gd(  p_lambda, p_M, p_weightMatrix )

load global_variables.mat;

p_lambda = 0.1;

chunkSize = ceil(totalTestRecords/(p_M-1));

mu = zeros(p_M, 46);

for i=2:p_M
    if((i*chunkSize)<totalTestRecords)
        mu(i, :) = mean(featuresMatrixTest((i-2)*chunkSize+1:i*chunkSize, :))*(i*1.3);
    else
        mu(i, :) = mean(featuresMatrixTest((i-2)*chunkSize+1:totalTestRecords,:))*(i*1.3);
    end
end

phiMatrix = zeros(totalTestRecords, p_M);
for i=1:totalTestRecords
    phiMatrix(i,1) = 1;
end
for j=2:p_M
    for i=1:totalTestRecords
        xMinusMu = (featuresMatrixTest(i,:)-mu(j,:));
        sigmaSquare = var(featuresMatrixTraining(i,:))*eye(numOfFeaturesTraining);
        phiMatrix(i,j) = exp(-((xMinusMu)*pinv(sigmaSquare)*transpose(xMinusMu)));
    end
end
 
    %Squared Error Function
    phiWeight = (phiMatrix*p_weightMatrix - relevanceMatrixTest);
    
    squaredError = transpose(phiWeight)*phiWeight + 0.5*p_lambda*transpose(p_weightMatrix)*p_weightMatrix;
    
    %Root Mean Square Error
    r_eRMS = sqrt((2*squaredError)/totalTestRecords);


end

