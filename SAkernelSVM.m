function [ error ] = SAkernelSVM( C, index_testing,...
    index_training, Labels, num_samples, size_training )
%SAkernelSVM is a capsule function for LIBSVM prediction using a
%   precomputed Sensing-Aware kernel.
%
% Created by Chengxi Yang on 12/19/14
%

%% Import data
load('kernel1_dvd_electronics.mat')
Labels=ones(4000,1);
Labels(1:2000)=-1;
% Labels(2001:3000)=-1;

%% Use LIBSVM with precomputed kernel for classification
accuracy = zeros(25,3); % 25 sub-experiments in total
for ii = 1:25
    % Train the model with training set
    model = svmtrain(Labels(index_training(ii,:)),[(1:size_training)',...
        kernel_total(index_training(ii,:),index_training(ii,:))],sprintf('-t 4 -c %f',C));
    % Test the model with testing set
    [~,accuracy(ii,:),~] = svmpredict(Labels(...
        index_testing(ii,:)),[(1:(num_samples-size_training))',...
        kernel_total(index_testing(ii,:),index_training(ii,:))],model);
end
error = 100-mean(accuracy(:,1));

end