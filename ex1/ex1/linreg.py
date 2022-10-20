import numpy as np


def load_data(path, num_train):
    """ Load the data matrices
    Input:
    path: string describing the path to a .csv file
          containing the dataset
    num_train: number of training samples
    Output:
    X_train: numpy array of shape num_train x 11
             containing the first num_train many
             data rows of columns 1 to 11 of the
             .csv file.
    Y_train: numpy array of shape num_train
             containing the first num_train many
             data rows of column 12 of the .csv
             file.
    X_test: same as X_train only corresponding to
            the remaining rows after the first 
            num_train many rows.
    Y_test: same as Y_train only corresponding to
            the remaining rows after the first 
            num_train many rows.
    """
    # TODO: load data according to the specifications,
    # e.g. using numpy.loadtxt

    
    X_train = np.loadtxt(path, delimiter=";",skiprows=1,usecols=np.arange(0, 11),max_rows=num_train)
    Y_train = np.loadtxt(path, delimiter=";",skiprows=1,usecols=(11),max_rows=num_train)
    X_test = np.loadtxt(path, delimiter=";",skiprows=num_train,usecols=np.arange(0, 11))
    Y_test = np.loadtxt(path, delimiter=";",skiprows=num_train,usecols=(11))

    return X_train, Y_train, X_test, Y_test


def fit(X, Y):
    """ Fit linear regression model
    Input:
    X: numpy array of shape N x n containing data
    Y: numpy array of shape N containing targets
    Output:
    theta: numpy array of shape n + 1 containing weights
           obtained by fitting data X to targets Y
           using linear regression
    """
    # TODO

    shape_X = np.shape(X)                       # get shape of X

    X = np.transpose(X)                         # transpose X
    X = np.vstack([X, np.ones(shape_X[0])])     # add 1 to the end of X
    X = np.transpose(X)                         # transpose X again

    theta  = np.linalg.lstsq(X,Y,rcond=None)    # use np function for linear regession
    theta = theta[0]                            # use only the 0 element of the return tuple of linalg.lstsq

    return theta


def predict(X, theta):
    """ Perform inference using data X
        and weights theta
    Input:
    X: numpy array of shape N x n containing data
    theta: numpy array of shape n + 1 containing weights
    Output:
    Y_pred: numpy array of shape N containig predictions
    """
    # TODO
    theta = np.transpose(theta)                 # transpose theta

    shape_X = np.shape(X)                       # get shape of X

    X = np.transpose(X)                         # transpose X
    X = np.vstack([X, np.ones(shape_X[0])])     # add 1 to the end of X
    X = np.transpose(X)                         # transpose X again

    Y_pred = X @ theta                          # Matrix multiplication to predict with data X an theta the value of Y_pred
    return Y_pred


def energy(Y_pred, Y_gt):
    """ Calculate squared error
    Input:
    Y_pred: numpy array of shape N containing prediction
    Y_gt: numpy array of shape N containing targets
    Output:
    se: squared error between Y_pred and Y_gt
    """
    # TODO

    se = np.sum((Y_pred-Y_gt)**2)             # sum over the squared error of the hole data

    return se
