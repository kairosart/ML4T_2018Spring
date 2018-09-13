
Skip to content

    Features
    Platform
    Business
    Explore
    Pricing

Sign in
Sign up

2
2

    5

ntrang086/ml_trading_defeat_learners
Code
Issues 0
Pull requests 0
Projects 0
Insights
Join GitHub today

GitHub is home to over 28 million developers working together to host and review code, manage projects, and build software together.
ml_trading_defeat_learners/DTLearner.py
3561cad on Jan 16
@ntrang086 ntrang086 fix: Revise the way we choose best feature to split on
236 lines (175 sloc) 8.72 KB
"""A simple wrapper for Decision Tree regression"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from copy import deepcopy
from collections import Counter
from operator import itemgetter


class DTLearner(object):

    def __init__(self, leaf_size=1, verbose=False, tree=None):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = deepcopy(tree)
        if verbose:
            self.get_learner_info()
        

    def __build_tree(self, dataX, dataY, rootX=[], rootY=[]):
        """Builds the Decision Tree recursively by choosing the best feature to split on and 
        the splitting value. The best feature has the highest absolute correlation with dataY. 
        If all features have the same absolute correlation, choose the first feature. The 
        splitting value is the median of the data according to the best feature. 
        If the best feature doesn't split the data into two groups, choose the second best 
        one and so on; if none of the features does, return leaf
        Parameters:
        dataX: A numpy ndarray of X values at each node
        dataY: A numpy 1D array of Y values at each node
        rootX: A numpy ndarray of X values at the parent/root node of the current one
        rootY: A numpy 1D array of Y values at the parent/root node of the current one
        
        Returns:
        tree: A numpy ndarray. Each row represents a node and four columns are feature indices 
        (int type; index for a leaf is -1), splitting values, and starting rows, from the current 
        root, for its left and right subtrees (if any)
        """
        # Get the number of samples (rows) and features (columns) of dataX
        num_samples = dataX.shape[0]
        num_feats = dataX.shape[1]

        # If there is no sample left, return the most common value from the root of current node
        if num_samples == 0:
            return np.array([-1, Counter(rootY).most_common(1)[0][0], np.nan, np.nan])

        # If there are <= leaf_size samples or all data in dataY are the same, return leaf
        if num_samples <= self.leaf_size or len(pd.unique(dataY)) == 1:
            return np.array([-1, Counter(dataY).most_common(1)[0][0], np.nan, np.nan])
    
        avail_feats_for_split = list(range(num_feats))

        # Get a list of tuples of features and their correlations with dataY
        feats_corrs = []
        for feat_i in range(num_feats):
            abs_corr = abs(pearsonr(dataX[:, feat_i], dataY)[0])
            feats_corrs.append((feat_i, abs_corr))
        
        # Sort the list in descending order by correlation
        feats_corrs = sorted(feats_corrs, key=itemgetter(1), reverse=True)

        # Choose the best feature, if any, by iterating over feats_corrs
        feat_corr_i = 0
        while len(avail_feats_for_split) > 0:
            best_feat_i = feats_corrs[feat_corr_i][0]
            best_abs_corr = feats_corrs[feat_corr_i][1]

            # Split the data according to the best feature
            split_val = np.median(dataX[:, best_feat_i])

            # Logical arrays for indexing
            left_index = dataX[:, best_feat_i] <= split_val
            right_index = dataX[:, best_feat_i] > split_val

            # If we can split the data into two groups, then break out of the loop            
            if len(np.unique(left_index)) != 1:
                break
            
            avail_feats_for_split.remove(best_feat_i)
            feat_corr_i += 1
        
        # If we complete the while loop and run out of features to split, return leaf
        if len(avail_feats_for_split) == 0:
            return np.array([-1, Counter(dataY).most_common(1)[0][0], np.nan, np.nan])

        # Build left and right branches and the root                    
        lefttree = self.__build_tree(dataX[left_index], dataY[left_index], dataX, dataY)
        righttree = self.__build_tree(dataX[right_index], dataY[right_index], dataX, dataY)

        # Set the starting row for the right subtree of the current root
        if lefttree.ndim == 1:
            righttree_start = 2 # The right subtree starts 2 rows down
        elif lefttree.ndim > 1:
            righttree_start = lefttree.shape[0] + 1
        root = np.array([best_feat_i, split_val, 1, righttree_start])

        return np.vstack((root, lefttree, righttree))
    

    def __tree_search(self, point, row):
        """A private function to be used with query. It recursively searches 
        the decision tree matrix and returns a predicted value for point
        Parameters:
        point: A numpy 1D array of test query
        row: The row of the decision tree matrix to search
    
        Returns 
        pred: The predicted value
        """

        # Get the feature on the row and its corresponding splitting value
        feat, split_val = self.tree[row, 0:2]
        
        # If splitting value of feature is -1, we have reached a leaf so return it
        if feat == -1:
            return split_val

        # If the corresponding feature's value from point <= split_val, go to the left tree
        elif point[int(feat)] <= split_val:
            pred = self.__tree_search(point, row + int(self.tree[row, 2]))

        # Otherwise, go to the right tree
        else:
            pred = self.__tree_search(point, row + int(self.tree[row, 3]))
        
        return pred


    def addEvidence(self, dataX, dataY):
        """Add training data to learner
        Parameters:
        dataX: A numpy ndarray of X values of data to add
        dataY: A numpy 1D array of Y training values
        Returns: An updated tree matrix for DTLearner
        """

        new_tree = self.__build_tree(dataX, dataY)

        # If self.tree is currently None, simply assign new_tree to it
        if self.tree is None:
            self.tree = new_tree

        # Otherwise, append new_tree to self.tree
        else:
            self.tree = np.vstack((self.tree, new_tree))
        
        # If there is only a single row, expand tree to a numpy ndarray for consistency
        if len(self.tree.shape) == 1:
            self.tree = np.expand_dims(self.tree, axis=0)
        
        if self.verbose:
            self.get_learner_info()
        
        
    def query(self, points):
        """Estimates a set of test points given the model we built
        
        Parameters:
        points: A numpy ndarray of test queries
        Returns: 
        preds: A numpy 1D array of the estimated values
        """

        preds = []
        for point in points:
            preds.append(self.__tree_search(point, row=0))
        return np.asarray(preds)


    def get_learner_info(self):
        print ("Info about this Decision Tree Learner:")
        print ("leaf_size =", self.leaf_size)
        if self.tree is not None:
            print ("tree shape =", self.tree.shape)
            print ("tree as a matrix:")
            # Create a dataframe from tree for a user-friendly view
            df_tree = pd.DataFrame(self.tree, columns=["factor", "split_val", "left", "right"])
            df_tree.index.name = "node"
            print (df_tree)
        else:
            print ("Tree has no data")


if __name__=="__main__":
    print ("This is a Decision Tree Learner\n")

    # Some data to test the DTLearner
    x0 = np.array([0.885, 0.725, 0.560, 0.735, 0.610, 0.260, 0.500, 0.320])
    x1 = np.array([0.330, 0.390, 0.500, 0.570, 0.630, 0.630, 0.680, 0.780])
    x2 = np.array([9.100, 10.900, 9.400, 9.800, 8.400, 11.800, 10.500, 10.000])
    x = np.array([x0, x1, x2]).T
    
    y = np.array([4.000, 5.000, 6.000, 5.000, 3.000, 8.000, 7.000, 6.000])

    # Create a tree learner from given train X and y
    dtl = DTLearner(verbose=True, leaf_size=1)
    print ("\nAdd data")
    dtl.addEvidence(x, y)

    print ("\nCreate another tree learner from an existing tree")
    dtl2 = DTLearner(tree=dtl.tree)

    # dtl2 should have the same tree as dtl
    assert np.any(dtl.tree == dtl2.tree)

    dtl2.get_learner_info()

    # Modify the dtl2.tree and assert that this doesn't affect dtl.tree
    dtl2.tree[0] = np.arange(dtl2.tree.shape[1])
    assert np.any(dtl.tree != dtl2.tree)

    # Query with dummy data
    dtl.query(np.array([[1, 2, 3], [0.2, 12, 12]]))

    # Another dataset to test that "If the best feature doesn't split the data into two
    # groups, choose the second best one and so on; if none of the features does, return leaf"
    x2 = np.array([
     [  0.26,    0.63,   11.8  ],
     [  0.26,    0.63,   11.8  ],
     [  0.32,    0.78,   10.   ],
     [  0.32,    0.78,   10.   ],
     [  0.32,    0.78,   10.   ],
     [  0.735,   0.57,    9.8  ],
     [  0.26,    0.63,   11.8  ],
     [  0.61,    0.63,    8.4  ]])
        
    y2 = np.array([ 8.,  8.,  6.,  6.,  6.,  5.,  8.,  3.])
        
    dtl = DTLearner(verbose=True)
    dtl.addEvidence(x2, y2)

