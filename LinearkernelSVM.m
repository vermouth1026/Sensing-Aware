function [ error ] = LinearkernelSVM( C, index_testing,...
    index_training, Labels, Feature_Matrix )
%LinearkernelSVM is a capsule function for LIBSVM prediction using
%   the Linear kernel.
%
% Created by Chengxi Yang on 12/19/14
%

accuracy = zeros(5,3);
for ii = 1:5
    % Train the model with training set
    model = svmtrain(Labels(index_training(ii,:)),Feature_Matrix(...
        index_training(ii,:),:),sprintf('-t 0 -c %f',...
        C*1e-3));
    % Test the model with testing set
    [~,accuracy(ii,:),~] = svmpredict(...
        Labels(index_testing(ii,:)),Feature_Matrix(index_testing(ii,...
        :),:),model);
end
error = 100 - mean(accuracy(:,1));

end