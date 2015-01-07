close all; clc; clear;

size_training = 800;
num_samples = 4000;
index_training = zeros(25,size_training);
index_testing = zeros(25,num_samples - size_training);

% Generate 5 splits
for seed = 1:5
    s = RandStream('mt19937ar','Seed',seed);
    RandStream.setGlobalStream(s);
    rand_perm = randperm(num_samples);
    for ii = 0:4
        % Randomly select 800 documents as traning set
        index_training(ii+5*seed-4,:) = rand_perm(ii*size_training+1:(ii+1)*...
            size_training);
        index_testing(ii+5*seed-4,:) = setdiff(rand_perm,index_training(ii+5*seed-4,:),...
            'stable');
    end
end

save('split_indices.mat','index_training','index_testing')