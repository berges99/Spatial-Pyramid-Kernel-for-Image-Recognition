'''

Spatial pyramid kernel implementation with Python.

Automates the process of reading the data and executes the SVM with the specified kernel parameters 
through the terminal.

__authors__ = David Bergés Lladó, Roser Cantenys Sabà and Alex Carrillo Alza

'''



#####################################################################################################
# IMPORTING LIBRARIES
#####################################################################################################

import argparse
import numpy as np
import pandas as pd

from time import time
from functools import wraps
from collections import Counter

from sklearn import svm
from sklearn.model_selection import train_test_split



#####################################################################################################
# UTILITY FUNCTIONS
#####################################################################################################

def print_duration(start, end):
    ''' Display the duration of an execution in the format -> 00:00:00.00
    ----------
    PARAMETERS
    - start, end: time.time() object representing CPU time at a certain moment
    ----------
    RETURNS
    - None

    '''
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}\n".format(int(hours), int(minutes), seconds))


def timing(f):
    ''' Decorator used to claculate the time used to compute a function
    ----------
    PARAMETERS
    - f: executable function
    ----------
    RETURNS
    - returns a wrapper for time calculation for the function f

    '''
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        # Display the elapsed time in pretty format -> 00:00:00.00
        print(f"Elapsed time for {f.__name__}(): ", end = "")
        print_duration(start, end)

        return result

    return wrapper


@timing
def read_data(data_path):
    ''' Read a .csv file from the specified path
    ----------
    PARAMETERS
    - data_path: string with the name of the file to be read

    (notice that data_path only contains the name of the file, and thus the file
    must be located in the same directory as this 'histogram_svm.py' script)
    ----------
    RETURNS
    - a pd.DataFrame with the representation of the data

    '''
    # Specify dtypes and column names
    length = int(data_path.split('.')[0][2:])**2
    dtypes = {'pixel-' + str(i): 'uint8' for i in range(1, length +1)}
    dtypes.update({'label' : 'category'})
    colnames = list(dtypes.keys())

    print('-' * 60)
    print(f"Reading {data_path}...")
    data = pd.read_csv(
        data_path, header = None, names = colnames, dtype = dtypes
    )
    # Output some metrics of the data file
    print(f"train.cv file has {data.shape[0]} rows and {data.shape[1]} columns")
    print(f"Memory usage: {round(data.memory_usage().sum() / 1024**2, 3)} Mb")
    print('-' * 60)
    
    return data


@timing
def quantization(data, n_bins):
    ''' Returns the dataframe with all the values quantized into n_bins levels.
    ---------
    PARAMETERS
    - data: pd.DataFrame to be quantized (without the response value 'target')
    - n_bins: integer representing the number of levels of the quantization
    ---------
    RETURNS
    - a pd.DataFrame with all the values quantized

    '''
    print('-' * 60)
    print(f"Quantizing into {n_bins} levels...")
    print('-' * 60)
    # Hence (as the colormap is grayspace) we only have 255 levels
    l = 255 // n_bins
    # Define the mapping ranges
    mapping = {range((i*l), (i*l + l)): i for i in range(n_bins)}
    # Apply the quantization elementwise in the dataframe
    return data.applymap(lambda x: next((v for k, v in mapping.items() if x in k), 0))



#####################################################################################################
# HISTOGRAM KERNEL FUNCTION
#####################################################################################################

def histogram_kernel_wrapper(L: int = 0):

    # Global wrapper variables (# parameters, and image resolution (res x res))
    length, res = 0, 0


    def divide_image(im, l):
        ''' Divide the image into equally sized blocks
        ----------
        PARAMETERS
        - im: flattened numpy.ndarray representing an image
        - l: integer representing the level of divisions that should be performed
        ----------
        RETURNS
        - an array with a flattened numpy.ndarray of every block at every position

        '''
        # Compute resolution of each block
        newres = res // (2**l)
        # Unflatten the image array to its original size
        im = np.reshape(im, newshape = (res, res))

        # Split the image into 2^2l sub-blocks
        splits = (im.reshape(res // newres, newres, -1, newres)
                    .swapaxes(1, 2)
                    .reshape(-1, newres, newres))

        # Return the flattened split of the image
        return [split.flatten() for split in splits]


    def compute_all_histograms(X):
        '''


        '''
        # We will have a total of L+1 different histograms
        hist = []
        for i in range(L + 1):
            hist_i = []

            # Compute the histogram for every image
            for j in range(X.shape[0]):
                # No splitting required if l = 0
                if i == 0:
                    hist_i.append(Counter(X[j]))
                # Else compute the image splits, and calculate the histogram for each block
                else:
                    splits = divide_image(X[j], i)
                    # For every block
                    for k in range(len(splits)):
                        splits[k] = Counter(splits[k])
                    hist_i.append(splits)

            # Append the histogram calculation of l = i to the global list
            hist.append(hist_i)
            
        return hist


    def K(h1, h2):
        ''' Compute the histogram-based-kernel value between two images
        ----------
        PARAMETERS
        - im1, im2: flattened numpy.ndarray representing an image of the collection
        - L: integer representing the level of divisions that should be performed
        ----------
        RETURNS
        - a float representing the value of the kernel between images 'im1' and 'im2'

        '''
        # For I_0
        k_ij = (1/2**L) * (sum((h1[0] & h2[0]).values()) / length)

        # For every level of partitioning (I_1, I_2, ..., I_L):
        for l in range(1, L + 1):

            # Factor in the l-th iteration
            factor = 1 / (2**(L - l + 1))

            # Compute and add histogram intersection of every block
            for k in range(len(h1[l])):
                k_ij += factor * (sum((h1[l][k] & h2[l][k]).values()) / length)

        return k_ij


    def hist_kernel(X, Y):
        ''' Histogram kernel function -> computes the Gram Matrix of the kernel
        ----------
        PARAMETERS
        - X,Y: numpy.ndarray representing the data
        (notice that while training, Y = X, and thus the Gram Matrix is symmetric)
        ----------
        RETURNS
        - Gram_matrix: numpy.ndarray representing the Gram_matrix of the histogram kernel

        '''
        # Update image resolution
        nonlocal length
        length = X.shape[1]
        nonlocal res
        res = int(np.sqrt(length))

        # Initialize the Gram matrix with zeros (allocate memory)
        Gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
        # If X == Y, i.e. we are training the SVM, the Gram Matrix will be symmetric, and
        # thus we can halve the total number of computations, as G[i,j] = G[j,i]
        if X.shape[0] == Y.shape[0]:
            # Construct the histogram matrices
            histograms = compute_all_histograms(X)
            for i in range(X.shape[0]):
                # Get all the histograms (for all particions) of image i
                h1 = [histograms[k][i] for k in range(L + 1)]
                for j in range(i, X.shape[0]):
                    # Get all the istograms (for all particions) of image j
                    h2 = [histograms[k][j] for k in range(L + 1)]
                    # Compute the intersection of image i and image j (histograms)
                    Gram_matrix[i, j] = K(h1, h2)
                    Gram_matrix[j, i] = Gram_matrix[i, j].copy()
        # Otherwise, the matrix is not symmetric, we cannot reuse computations
        else:
            # Construct the histogram matrices
            # Hence that they are not the same if we are not in training
            histogramsX = compute_all_histograms(X)
            histogramsY = compute_all_histograms(Y)
            for i in range(X.shape[0]):
                # Get all the histograms (for all particions) for image i
                h1 = [histogramsX[k][i] for k in range(L + 1)]
                for j in range(Y.shape[0]):
                    # Get all the histograms (for all particions) for image j
                    h2 = [histogramsY[k][j] for k in range(L + 1)]
                    # Compute the intersection of image i and image j (histograms)
                    Gram_matrix[i, j] = K(h1, h2)

        return Gram_matrix


    return hist_kernel



#####################################################################################################
# MAIN PROGRAM (fit and predict with the SVM)
#####################################################################################################

if __name__ == '__main__':

    # Parser arguments (arguments taken by the script)
    parser = argparse.ArgumentParser(
        description = 'Compute the histogram kernel SVM.')
    parser.add_argument(
        '--data', type = str, help = 'File from where to read image data.')
    parser.add_argument(
        '--quantization', type = int, default = None, help = 'Quantization level for the histograms.')
    parser.add_argument(
        '--L', type = int, default = 3, help = 'Level of image splitting in the kernel.')
    parser.add_argument(
        '--train_frac', type = float, default = .25,
        help = 'Fraction of train samples to be taken from data.')
    parser.add_argument(
        '--test_frac', type = float, default = .1,
        help = 'Fraction of test samples to be taken from data.')

    # Get the arguments
    args = parser.parse_args()
    data_path = args.data                                       # path to data file
    n_bins = args.quantization; L = args.L                      # kernel parameters
    train_frac = args.train_frac; test_frac = args.test_frac    # train_test splitting

    print(f"Histogram-kernel SVM with L = {L} and quantization = {n_bins}")
    print(f"train size = {6379 // int(train_frac**(-1))}, test size = {6379 // int(test_frac**(-1))}")    

    # Read the data
    df = read_data(data_path)
    # Split the target (label) from data
    Y = np.array(df.label)
    df = df.drop(columns = ['label'])

    # Check if quantization is required
    if n_bins is not None:
        df = quantization(df, n_bins)

    # Convert data to a numpy ndarray
    X = np.array(df).astype('uint8')

    # Train and test splitting
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, shuffle = True, stratify = Y, train_size = train_frac,
        test_size = test_frac, random_state = 347
    )


    ###
    # Fit the SVM
    ###

    # Initialize an instance of a histogram-kernel SVM with specified L
    cvf = svm.SVC(kernel = histogram_kernel_wrapper(L = L))

    print('-' * 60)

    print('Fitting the SVM with training data...')
    start = time()
    # Train the SVM with the training samples
    cvf.fit(x_train, y_train)
    end = time()
    print('Elapsed time for training the SVM: ', end = '')
    print_duration(start, end)

    
    print('Predicting test values...')
    start = time()
    # Predict on the test set
    test_predictions = cvf.predict(x_test)
    end = time()
    print('Elapsed time for predicting on test data: ', end = '')
    print_duration(start, end)

    print()
    print('-' * 60)
    # Output the accuracy on the test set
    print(f"Test accuracy: {sum(test_predictions == y_test) / len(y_test)}")
    