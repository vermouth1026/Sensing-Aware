%
% Created by Chengxi Yang on 12/19/14
%

close all; clc; clear;

%% Prepare the data
load('movie_dvd_data.mat')

Labels=ones(4000,1);
Labels(1:2000)=-1;
% Labels(2001:3000)=-1;
num_words = size(Feature_Matrix,2);
num_samples = size(Labels,1);
size_training = num_samples/5;

% Average length of each document
avelen = floor(sum(sum(Feature_Matrix))/4000);

% Import the splits
load('split_indices.mat')

%% Generate Sensing-Aware Kernel Matrices

% kernel_train = zeros(size_training,size_training,5);
% for kk = 1:5
%     for ii = 1:size_training
%         for jj = 1:size_training
%             kernel_train(ii,jj,kk) = ker_value_sensing1(...
%                 Feature_Matrix(index_training(kk,ii),:),...
%                 Feature_Matrix(index_training(kk,jj),:),num_words,avelen);
%         end
%     end
% end
% 
% n = num_samples - size_training;
% kernel_test = zeros(n,size_training,5);
% for kk = 1:5
%     for ii = 1:size_training
%         for jj = 1:n
%             kernel_test(jj,ii,kk) = ker_value_sensing1(...
%                 Feature_Matrix(index_training(kk,ii),:),...
%                 Feature_Matrix(index_testing(kk,jj),:),num_words,avelen);
%         end
%     end
% end
% save('kernel1_books_dvd_sent.mat','kernel_train','kernel_test')

% % Patch for kernel matrix generation
% load('kernel1_movie_dvd.mat')
% for i=1:5
% kernel_total(index_training(i,:),index_training(i,:))=kernel_train(:,:,i);
% kernel_total(index_testing(i,:),index_training(i,:))=kernel_test(:,:,i);
% end
% save('kernel1_movie_dvd.mat','kernel_total')

%% Tune the parameters and find maximal accuracy
% RBF kernel
ObjectiveFunction = @(x) RBFkernelSVM(x, index_testing,...
    index_training, Labels, Feature_Matrix);
X0 = [2 0.8]; % scaled for better precision
[C,error] = simulannealbnd(ObjectiveFunction,X0,...
    [1 0.01],[5 1.5]);
accuracy = 100 - error;
disp(accuracy)

% % Sensing-Aware kernel
% ObjectiveFunction = @(x) SAkernelSVM(x, index_testing,...
%     index_training, Labels, num_samples, size_training);
% X0 = 0.1;
% [C,error] = simulannealbnd(ObjectiveFunction,X0,...
%     0.01,1);
% accuracy = 100 - error;
% disp(accuracy)