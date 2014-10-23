function [ r_optimalLambda, r_minERMS ] = validate_cfs( p_weightMatrix, p_M )

load global_variables.mat;

lambda = 0.1;
r_minERMS = 10000; % Have tested it'll be well below this value

chunkSize = ceil(totalValidationRecords/(p_M-1));

mu = zeros(p_M, 46);

for i=2:p_M
    if((i*chunkSize)<totalValidationRecords)
        mu(i, :) = mean(featuresMatrixValidation((i-2)*chunkSize+1:(i-1)*chunkSize, :))*(i*1.3);
    else
        mu(i, :) = mean(featuresMatrixValidation((i-2)*chunkSize+1:totalValidationRecords,:))*(i*1.3);
    end
end

phiMatrix = zeros(totalValidationRecords, p_M);

for i=1:totalValidationRecords
    phiMatrix(i,1) = 1;
end
for j=2:p_M
    for i=1:totalValidationRecords
        xMinusMu = (featuresMatrixValidation(i,:)-mu(j,:));
        sigmaSquare = var(featuresMatrixTraining(i,:))*eye(numOfFeaturesTraining);
        phiMatrix(i,j) = exp(-((xMinusMu)*pinv(sigmaSquare)*transpose(xMinusMu)));
    end
end

while lambda < 100
    
    %Squared Error Function
    phiWeight = (phiMatrix*p_weightMatrix - relevanceMatrixValidation);
    
    squaredError = transpose(phiWeight)*phiWeight + 0.5*lambda*transpose(p_weightMatrix)*p_weightMatrix;
    
    %Root Mean Square Error
    eRMS = sqrt((2*squaredError)/totalValidationRecords);
    
    if(eRMS < r_minERMS)
        r_optimalLambda = lambda;
        r_minERMS = eRMS;
    end
    
    lambda = lambda * 10;
end

end

