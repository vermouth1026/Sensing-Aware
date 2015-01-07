function [ error ] = RBFkernelSVM( C, index_testing,...
    index_training, Labels, Feature_Matrix )
%RBFkernelSVM is a capsule function for LIBSVM prediction using
%   the RBF kernel.
%
% Created by Chengxi Yang on 12/19/14
%

accuracy = zeros(25,3);
for ii = 1:25
    % Train the model with training set
    model = svmtrain(Labels(index_training(ii,:)),Feature_Matrix(...
        index_training(ii,:),:),sprintf('-t 2 -c %f -g %f',...
        C(1)*1e3,C(2)/1e5)); % scaled for better precision
    % Test the model with testing set
    [~,accuracy(ii,:),~] = svmpredict(...
        Labels(index_testing(ii,:)),Feature_Matrix(index_testing(ii,...
        :),:),model);
end
error = 100 - mean(accuracy(:,1));

end