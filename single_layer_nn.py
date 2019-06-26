import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
from time import time
import pandas as pd
import matplotlib.pyplot as plt
import pickle


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    sigmoid_op = 1.0 / (1.0 + np.exp(-1.0 * z))

    return sigmoid_op


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples.
    n_train_samples = 50000
    n_data_features = 784
    train_data_preprocess = np.zeros((0, n_data_features))
    train_label_preprocess = []
    test_data = np.zeros((0, n_data_features))
    test_label = []

    # Create train and test datasets, and train and test labels
    for i in range(10):
        train_data_preprocess = np.concatenate(
            (train_data_preprocess, mat['train' + str(i)]), 0)
        train_label_preprocess = np.concatenate(
            (train_label_preprocess, np.ones(mat['train' + str(i)].shape[0]) * i), 0)
        test_data = np.concatenate(
            (test_data, mat['test' + str(i)]), 0)
        test_label = np.concatenate(
            (test_label, np.ones(mat['test' + str(i)].shape[0]) * i), 0)

    # Save labels as integers
    train_label_preprocess = train_label_preprocess.astype(int)
    test_label = test_label.astype(int)

    # Normalize the data
    train_data_preprocess = np.double(train_data_preprocess) / 255.0
    test_data = np.double(test_data) / 255.0

    # Split train data into train and validation sets
    shuffled_row_numbers = np.random.permutation(
        range(train_data_preprocess.shape[0]))
    train_data = train_data_preprocess[shuffled_row_numbers[0:n_train_samples], :]
    validation_data = train_data_preprocess[shuffled_row_numbers[n_train_samples:], :]

    # Use same permutation of numbers for train and test labels
    train_label = train_label_preprocess[shuffled_row_numbers[0:n_train_samples]]
    validation_label = train_label_preprocess[shuffled_row_numbers[n_train_samples:]]

    # Feature selection
    irrelevant_features = []
    for i in range(n_data_features):
        if (np.ptp(train_data_preprocess[:, i]) == 0.0 and np.ptp(test_data[:, i]) == 0.0):
            irrelevant_features.append(i)

    selected_features_set = set(list(range(0, 784))) - set(irrelevant_features)
    for feature in selected_features_set:
        selected_features.append(feature)

    train_data = np.delete(train_data, irrelevant_features, axis=1)
    validation_data = np.delete(validation_data, irrelevant_features, axis=1)
    test_data = np.delete(test_data, irrelevant_features, axis=1)

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0.0

    # Gradient for input weights and output weights
    obj_grad_w1 = 0.0
    obj_grad_w2 = 0.0

    num_samples = training_data.shape[0]

    # Save true outputs
    true_outputs = np.zeros((num_samples, n_class))
    for i in range(num_samples):
        true_outputs[i][training_label[i]] = 1

    # Feedforward propagation

    # Propagating from input layer to hidden layer
    linear_comb_input = np.dot(np.column_stack(
        (training_data, np.ones(num_samples))), w1.T)
    output_hidden = sigmoid(linear_comb_input)
    # Propagating from hidden layer to output layer
    linear_comb_output = np.dot(np.column_stack(
        (output_hidden, np.ones(output_hidden.shape[0]))), w2.T)
    output_final = sigmoid(linear_comb_output)

    # Finding error and backpropagating

    # Error function
    obj_val = true_outputs * \
        np.log(output_final) + (1 - true_outputs) * np.log(1 - output_final)
    obj_val = (-1/num_samples)*(np.sum(obj_val[:, :]))

    # Find gradients

    # Finding gradient of error between hidden layer and output layer
    delta_output = output_final - true_outputs
    obj_grad_w2 = np.dot(delta_output.T, np.column_stack(
        (output_hidden, np.ones(output_hidden.shape[0]))))

    # Finding gradient of error between input layer and hidden layer
    obj_grad_w1 = np.dot(np.dot(w2.T, delta_output.T)*(((np.column_stack(
        (output_hidden, np.ones(output_hidden.shape[0])))).T)*(1-(np.column_stack(
            (output_hidden, np.ones(output_hidden.shape[0])))).T)), np.column_stack(
        (training_data, np.ones(num_samples))))

    # Remove bias values, as gradient is not calculated for bias
    obj_grad_w1 = obj_grad_w1[:-1, :]

    # Calculate regularized error
    obj_val += (lambdaval * (np.sum(np.sum(w1**2)) +
                             np.sum(np.sum(w2**2)))) / (2.0*num_samples)

    # Regularization gradients
    obj_grad_w1 = (obj_grad_w1 + lambdaval * w1) / num_samples
    obj_grad_w2 = (obj_grad_w2 + lambdaval * w2) / num_samples

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.concatenate(
        (obj_grad_w1.flatten(), obj_grad_w2.flatten()), 0)

    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image

    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])
    num_samples = data.shape[0]
    # Propagating from input layer to hidden layer
    linear_comb_input = np.dot(np.column_stack(
        (data, np.ones(num_samples))), w1.T)
    output_hidden = sigmoid(linear_comb_input)
    # Propagating from hidden layer to output layer
    linear_comb_output = np.dot(np.column_stack(
        (output_hidden, np.ones(output_hidden.shape[0]))), w2.T)
    output_final = sigmoid(linear_comb_output)
    labels = np.argmax(output_final, axis=1)
    return labels


"""**************Neural Network Script Starts here********************************"""

# List to save irrelevant features, for feature selection
selected_features = []

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network
lambdas = [0, 10, 20, 30, 40, 50, 60]
hidden_nodes_nums = [4, 8, 12, 16, 20]
eval_results = []
for l in lambdas:
    for h in hidden_nodes_nums:

        start_time = time()

        # set the number of nodes in input unit (not including bias unit)
        n_input = train_data.shape[1]

        # set the number of nodes in hidden unit (not including bias unit)
        n_hidden = h  # 50

        # set the number of nodes in output unit
        n_class = 10

        # initialize the weights into some random matrices
        initial_w1 = initializeWeights(n_input, n_hidden)
        initial_w2 = initializeWeights(n_hidden, n_class)

        # unroll 2 weight matrices into single column vector
        initialWeights = np.concatenate(
            (initial_w1.flatten(), initial_w2.flatten()), 0)

        # set the regularization hyper-parameter
        lambdaval = l  # 0

        args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

        # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

        opts = {'maxiter': 50}  # Preferred value.

        nn_params = minimize(nnObjFunction, initialWeights, jac=True,
                             args=args, method='CG', options=opts)

        # In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
        # and nnObjGradient. Check documentation for this function before you proceed.
        # nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)

        # Reshape nnParams from 1D vector into w1 and w2 matrices
        w1 = nn_params.x[0:n_hidden *
                         (n_input + 1)].reshape((n_hidden, (n_input + 1)))
        w2 = nn_params.x[(n_hidden * (n_input + 1))
                          :].reshape((n_class, (n_hidden + 1)))

        # Test the computed parameters

        predicted_label = nnPredict(w1, w2, train_data)

        # find the accuracy on Training Dataset
        training_accuracy = 100 * \
            np.mean((predicted_label == train_label).astype(float))
        print('Training set Accuracy: ' + str(training_accuracy) + '%')

        predicted_label = nnPredict(w1, w2, validation_data)

        # find the accuracy on Validation Dataset
        validation_accuracy = 100 * \
            np.mean((predicted_label == validation_label).astype(float))
        print('Validation set Accuracy: ' + str(validation_accuracy) + '%')

        predicted_label = nnPredict(w1, w2, test_data)

        # find the accuracy on Validation Dataset
        test_accuracy = 100 * \
            np.mean((predicted_label == test_label).astype(float))
        print('Test set Accuracy: ' + str(test_accuracy) + '%')

        end_time = time()
        print('Time taken: ' + str(end_time-start_time) + 's')
        print('\n')
        eval_results.append([l, h, w1, w2, training_accuracy,
                             validation_accuracy, test_accuracy, end_time-start_time])

eval_results_df = pd.DataFrame(eval_results, columns=[
                               'Lambda', 'Hidden nodes', 'W1', 'W2', 'Train Accuracy', 'Validation Accuracy', 'Test Accuracy', 'Time'])


# Analyse the results

# Lambda vs Validation accuracy for 20 hidden nodes
df_hidden20 = eval_results_df[eval_results_df['Hidden nodes'] == 20]
print(df_hidden20)
plt.plot(df_hidden20['Lambda'], df_hidden20['Validation Accuracy'], '-o')
plt.xlabel('Lambda')
plt.ylabel('Validation Accuracy (%)')
plt.title('Lambda vs Validation Accuracy for 20 hidden nodes')
plt.show()

# Variation of accuracies (mean values) with changing lambda values
mean_accuracy_lambda = eval_results_df.groupby("Lambda").agg(
    {"Train Accuracy": np.mean, "Validation Accuracy": np.mean, "Test Accuracy": np.mean})
print(mean_accuracy_lambda)
plt.plot(mean_accuracy_lambda, '-o')
plt.legend(['Train Accuracy', 'Validation Accuracy', 'Test Accuracy'])
plt.xlabel('Lambda')
plt.ylabel('Accuracy (%)')
plt.title('Lambda vs accuracy')
plt.show()

# Lambda vs time
mean_time_lambda = eval_results_df.groupby(
    eval_results_df['Lambda']).agg({"Time": np.mean})
print(mean_time_lambda)
plt.plot(mean_time_lambda, '-o')
plt.xlabel('Lambda')
plt.ylabel('Time (in seconds)')
plt.title('Lambda vs time')
plt.show()

# Hidden nodes vs Validation Accuracy
mean_accuracy_hnodes = eval_results_df.groupby(
    "Hidden nodes").agg({"Validation Accuracy": np.mean})
print(mean_accuracy_hnodes)
plt.plot(mean_accuracy_hnodes, '-o')
plt.xlabel('Number of hidden nodes')
plt.ylabel('Validation Accuracy (%)')
plt.title('Hidden nodes vs Validation Accuracy')
plt.show()

# Variation of accuracies (mean values) with changing number of hidden nodes
mean_accuracies_hnodes = eval_results_df.groupby("Hidden nodes").agg(
    {"Train Accuracy": np.mean, "Validation Accuracy": np.mean, "Test Accuracy": np.mean})
print(mean_accuracies_hnodes)
plt.plot(mean_accuracies_hnodes, '-o')
plt.legend(['Train Accuracy', 'Validation Accuracy', 'Test Accuracy'])
plt.xlabel('Number of hidden nodes')
plt.ylabel('Accuracy (%)')
plt.title('Hidden nodes vs accuracy')
plt.show()

# Hidden nodes vs time
mean_hnodes = eval_results_df.groupby("Hidden nodes").agg({"Time": np.mean})
print(mean_hnodes)
plt.plot(mean_hnodes, '-o')
plt.xlabel('Number of hidden nodes')
plt.ylabel('Time (in seconds)')
plt.title('Hidden nodes vs time')
plt.show()

optimal_hyperparameters = eval_results_df.loc[(eval_results_df['Lambda'] == 30) & (
    eval_results_df['Hidden nodes'] == 20), ['Lambda', 'Hidden nodes', 'W1', 'W2']]

optimal_lambda, optimal_hnodes, optimal_w1, optimal_w2 = optimal_hyperparameters[
    'Lambda'], optimal_hyperparameters['Hidden nodes'], optimal_hyperparameters['W1'], optimal_hyperparameters['W2']

# Since these variables were stored in a pandas dataframe, these commands are used to properly
# access and make use of the hyperparameters
optimal_lambda=int(optimal_lambda)
optimal_hnodes=int(optimal_hnodes)
optimal_w1=list(optimal_w1)[0]
optimal_w2=list(optimal_w2)[0]

# Plot test accuracy when number of hidden nodes is optimal
plt.plot(df_hidden20['Lambda'],df_hidden20['Test Accuracy'],'-o')
plt.xlabel('Lambda')
plt.ylabel('Test Accuracy (%)')
plt.title('Lambda vs Test Accuracy for 20 hidden nodes')
plt.show()

# Plot test accuracy when lambda is optimal
df_lambda30=eval_results_df[eval_results_df['Lambda']==30]
print(df_lambda30)
plt.plot(df_lambda30['Hidden nodes'],df_lambda30['Test Accuracy'],'-o')
plt.xlabel('Number of Hidden Nodes')
plt.ylabel('Test Accuracy (%)')
plt.title('Lambda vs Test Accuracy for Lambda 30')
plt.show()

# Save selected features and optimal hyperparameters
with open("params.pickle", "wb") as f:
    pickle.dump((selected_features, optimal_lambda,
                 optimal_hnodes, optimal_w1, optimal_w2), f)

# Load parameters from params.pickle file
# selected_features, optimal_lambda, optimal_hnodes, optimal_w1, optimal_w2 = pickle.load(open('params.pickle', 'rb'))
