"""A simple wrapper for linear regression"""

import numpy as np

class LinRegLearner(object):

    def __init__(self, verbose = False):
        pass

    def addEvidence(self, dataX, dataY):
        """Add training data to learner
        Parameters:
        dataX: X values of data to add
        dataY: The Y training values
        """

        # Add 1s column so linear regression finds a constant term
        newdataX = np.ones([dataX.shape[0], dataX.shape[1] + 1])
        newdataX[:,0:dataX.shape[1]] = dataX

        # build and save the model
        self.model_coefs, residuals, rank, s = np.linalg.lstsq(newdataX, dataY)
        
    def query(self, points):
        """Estimate a set of test points given the model we built
        Parameters:
        points: A numpy array with each row corresponding to a specific query
        Returns: the estimated values according to the saved model
        """
        return (self.model_coefs[:-1] * points).sum(axis=1) + self.model_coefs[-1]

if __name__=="__main__":
print ("This is a Linear Regression Learner")