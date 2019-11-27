import numpy as np
import pandas as pd

from time import time
from functools import wraps
from collections import Counter

from sklearn import svm
from sklearn.model_selection import train_test_split


###


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


###


if __name__ == '__main__':

    print(f'Radial-basis kernel SVM for the 64x64 lego pieces dataset:')

    df = read_data('df64.csv')
    # Split the target (label) from data
    Y = np.array(df.label)
    df = df.drop(columns = ['label'])

    # Convert data to a numpy ndarray
    X = np.array(df).astype('uint8')

    # Train and test splitting
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, shuffle = True, stratify = Y, train_size = .5,
        test_size = .2, random_state = 347
    )

    ###
    # Fit the SVM
    ###

    # Initialize an instance of a histogram-kernel SVM with specified L
    cvf = svm.SVC(kernel = 'rbf', gamma = 'scale')

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



