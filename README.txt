1. Generate_Splits.m generates all the indices needed for 5 splits (i.e. which documents would be in the training set).
2. SAkernelSVM.m is a function that does training and prediction for sensing-aware kernel in one function. It returns the constant C and the overall prediction error.
3. RBFkernelSVM.m is a function that does training and prediction for RBF kernel in one function. It returns the overall prediction error and a vector C, in which C(1) is the constant C and C(2) is gamma.
4. ker_sensing_aware.m are functions that computes the corresponding sensing-aware kernel function value of two input vectors.
5. main.m consists of two main parts. The first half computes the sensing-aware kernel matrix for one dataset. The second half does the parameter tuning using simulated annealing and produces maximal prediction accuracy.

Notes:
All code are currently configured so that there are five 5-fold splits on each 4000 document dataset. Please check throughout the entire code to change the settings.
LIBSVM source code is not included in this package.