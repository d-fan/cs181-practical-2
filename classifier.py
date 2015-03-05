## This file provides starter code for extracting features from the xml files and
## for doing some learning.
##
## The basic set-up: 
## ----------------
## main() will run code to extract features, learn, and make predictions.
## 
## extract_feats() is called by main(), and it will iterate through the 
## train/test directories and parse each xml file into an xml.etree.ElementTree, 
## which is a standard python object used to represent an xml file in memory.
## (More information about xml.etree.ElementTree objects can be found here:
## http://docs.python.org/2/library/xml.etree.elementtree.html
## and here: http://eli.thegreenplace.net/2012/03/15/processing-xml-in-python-with-elementtree/)
## It will then use a series of "feature-functions" that you will write/modify
## in order to extract dictionaries of features from each ElementTree object.
## Finally, it will produce an N x D sparse design matrix containing the union
## of the features contained in the dictionaries produced by your "feature-functions."
## This matrix can then be plugged into your learning algorithm.
##
## The learning and prediction parts of main() are largely left to you, though
## it does contain code that randomly picks class-specific weights and predicts
## the class with the weights that give the highest score. If your prediction
## algorithm involves class-specific weights, you should, of course, learn 
## these class-specific weights in a more intelligent way.
##
## Feature-functions:
## --------------------
## "feature-functions" are functions that take an ElementTree object representing
## an xml file (which contains, among other things, the sequence of system calls a
## piece of potential malware has made), and returns a dictionary mapping feature names to 
## their respective numeric values. 
## For instance, a simple feature-function might map a system call history to the
## dictionary {'first_call-load_image': 1}. This is a boolean feature indicating
## whether the first system call made by the executable was 'load_image'. 
## Real-valued or count-based features can of course also be defined in this way. 
## Because this feature-function will be run over ElementTree objects for each 
## software execution history instance, we will have the (different)
## feature values of this feature for each history, and these values will make up 
## one of the columns in our final design matrix.
## Of course, multiple features can be defined within a single dictionary, and in
## the end all the dictionaries returned by feature functions (for a particular
## training example) will be unioned, so we can collect all the feature values 
## associated with that particular instance.
##
## Two example feature-functions, first_last_system_call_feats() and 
## system_call_count_feats(), are defined below.
## The first of these functions indicates what the first and last system-calls 
## made by an executable are, and the second records the total number of system
## calls made by an executable.
##
## What you need to do:
## --------------------
## 1. Write new feature-functions (or modify the example feature-functions) to
## extract useful features for this prediction task.
## 2. Implement an algorithm to learn from the design matrix produced, and to
## make predictions on unseen data. Naive code for these two steps is provided
## below, and marked by TODOs.
##
## Computational Caveat
## --------------------
## Because the biggest of any of the xml files is only around 35MB, the code below 
## will parse an entire xml file and store it in memory, compute features, and
## then get rid of it before parsing the next one. Storing the biggest of the files 
## in memory should require at most 200MB or so, which should be no problem for
## reasonably modern laptops. If this is too much, however, you can lower the
## memory requirement by using ElementTree.iterparse(), which does parsing in
## a streaming way. See http://eli.thegreenplace.net/2012/03/15/processing-xml-in-python-with-elementtree/
## for an example. 

import os
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse
import extractors
from extractors import ffs
from numpy import matlib, exp
import matplotlib.pyplot as plt
import sklearn.linear_model
import pickle
import util
import sys

def extract_feats(ffs, direc="train", global_feat_dict=None, silent=True):
    """
    arguments:
      ffs are a list of feature-functions.
      direc is a directory containing xml files (expected to be train or test).
      global_feat_dict is a dictionary mapping feature_names to column-numbers; it
      should only be provided when extracting features from test data, so that 
      the columns of the test matrix align correctly.

    returns: 
      a sparse design matrix, a dict mapping features to column-numbers,
      a vector of target classes, and a list of system-call-history ids in order 
      of their rows in the design matrix.
      
      Note: the vector of target classes returned will contain the true indices of the
      target classes on the training data, but will contain only -1's on the test
      data
    """
    fds = [] # list of feature dicts
    classes = []
    ids = [] 
    directory = os.listdir(direc)
    file_count = len(directory)
    for index, datafile in enumerate(directory):
        if not silent and index % 100 == 0:
            print "   Extracted %d of %d" % (index, file_count)
        # extract id and true class (if available) from filename
        # Keep it clazzy
        id_str,clazz = datafile.split('.')[:2]
        ids.append(id_str)
        # add target class if this is training data
        try:
            classes.append(util.malware_classes.index(clazz))
        except ValueError:
            # we should only fail to find the label in our list of malware classes
            # if this is test data, which always has an "X" label
            assert clazz == "X"
            classes.append(-1)
        rowfd = {}
        # parse file as an xml document
        tree = ET.parse(os.path.join(direc,datafile))
        # accumulate features
        [rowfd.update(ff(tree)) for ff in ffs]
        #print rowfd
        fds.append(rowfd)
        
    X,feat_dict = make_design_mat(fds,global_feat_dict)
    return X, feat_dict, np.array(classes), ids


def make_design_mat(fds, global_feat_dict=None):
    """
    arguments:
      fds is a list of feature dicts (one for each row).
      global_feat_dict is a dictionary mapping feature_names to column-numbers; it
      should only be provided when extracting features from test data, so that 
      the columns of the test matrix align correctly.
       
    returns: 
        a sparse NxD design matrix, where N == len(fds) and D is the number of
        the union of features defined in any of the fds 
    """
    if global_feat_dict is None:
        all_feats = set()
        [all_feats.update(fd.keys()) for fd in fds]
        feat_dict = dict([(feat, i) for i, feat in enumerate(sorted(all_feats))])
    else:
        feat_dict = global_feat_dict
        
    cols = []
    rows = []
    data = []        
    for i in xrange(len(fds)):
        temp_cols = []
        temp_data = []
        for feat,val in fds[i].iteritems():
            try:
                # update temp_cols iff update temp_data
                temp_cols.append(feat_dict[feat])
                temp_data.append(val)
            except KeyError as ex:
                if global_feat_dict is not None:
                    pass  # new feature in test data; nbd
                else:
                    raise ex

        # all fd's features in the same row
        k = len(temp_cols)
        cols.extend(temp_cols)
        data.extend(temp_data)
        rows.extend([i]*k)

    assert len(cols) == len(rows) and len(rows) == len(data)
   

    X = sparse.csr_matrix((np.array(data),
                   (np.array(rows), np.array(cols))),
                   shape=(len(fds), len(feat_dict)))
    return X, feat_dict
    
def sigma(clazz, features, weights):
    """
    Gives the probability that an observation with 'feature' is of class
    'clazz', given the weights.

    Both arguments should be numpy arrays
    """
    # w_k = weights[clazz]
    # phi = features
    # a_k = w_k . phi = dot(wegihts[clazz], features)
    
    # Sum of exp(a_j) = Sum of exp(w_k . phi)
    denominator = sum([exp(weights[k] * features.T) 
        for k in xrange(len(weights))])
    #print "Denominator", denominator

    #print "Numerator", exp(weights[clazz] * features.T)
    #      exp(a_k) = exp(w_k . phi)
    ret = exp(weights[clazz] * features.T) / denominator
    #print "Ret: ", ret
    return ret

def logistic(features, targets, weights, max_iter = 1000):
    """
    DO NOT USE THIS!!!! IT'S NOT DONE
    Uses Newton-Raphson to minimize error

    All arguments should be numpy matrices
    """
    assert(type(features) == np.matrix)
    assert(type(targets)  == np.matrix)
    assert(type(weights)  == np.matrix)
    assert(len(features)  == len(targets))
    assert(len(features[0]) == len(weights[0]))

    # y_n = sigma_n(w . phi)
    # R_nn = y_n * (1 - y_n)
    # grad E(W) = Phi . Phi . W - Phi . T = Phi . (Y - T)
    # H = Phi . Phi = Phi . R . Phi
    # 
    # W = (Phi . R . Phi)^-1 . Phi . R . (Phi . W - R^-1 (Y - T))
    
    # Number of features
    N = len(features)
    # Initialize R to N x N array of 0s, Y to N 0s
    R = matlib.zeros((N,N))
    Y = matlib.zeros((N,1))     # Column vector

    for i in xrange(max_iter):
        for n in xrange(2):
            print "Y:", Y
            Y[n] = sigma(n, features[n], weights)
            R[n,n] = Y[n] * (1-Y[n])
        print R

        # Bishop 4.100: Z = Phi . W - R^-1 (Y - T)
        Z = features * weights.T - R**-1 * (Y.T - targets)
        weights = (features.T * R * features)**-1 * features * R * Z

    pass

def sk_logistic(features, targets, regularization = 0.001):
    """
    Use Scikit Learn 'cause I'm lazy
    First return value is a function that returns a class number based on the input vector
    Second is the actual model objet
    """

    logreg = sklearn.linear_model.LogisticRegression(C=regularization)
    logreg.fit(features, targets)

    def predictor(feat):
        return logreg.predict(feat)[0]
    return predictor, logreg

## The following function does the feature extraction, learning, and prediction
def main(load = False):
    train_dir = "train"
    test_dir = "test"
    outputfile = "mypredictions.csv"  # feel free to change this or take it as an argument
    
    if not load:
        # extract features
        print "extracting training features..."
        X_train,global_feat_dict,t_train,train_ids = extract_feats(ffs, train_dir)
        print "done extracting training features"
        print
        print "Saving features"
        with open("X_train", "w") as out:
            pickle.dump(X_train, out)
        with open("global_feat_dict", "w") as out:
            pickle.dump(global_feat_dict, out)
        with open("t_train", "w") as out:
            pickle.dump(t_train, out)
        with open("train_ids", "w") as out:
            pickle.dump(train_ids, out)
        print "Done saving"
        print
    else:
        print "Loading previous features"
        with open("X_train", "r")           as out: X_train =           pickle.load(out)
        with open("global_feat_dict", "r")  as out: global_feat_dict =  pickle.load(out)
        with open("t_train", "r")           as out: t_train =           pickle.load(out)
        with open("train_ids", "r")         as out: train_ids =         pickle.load(out)
        print "Done loading"
        print
    
    # TODO train here, and learn your classification parameters
    print "learning..."
    predictor, _ = sk_logistic(X_train, t_train)
    # Start with logistic regression
    print "done learning"
    print
    
    # get rid of training data and load test data
    #del X_train
    #del t_train
    #del train_ids
    #print "extracting test features..."
    #X_test,_,t_ignore,test_ids = extract_feats(ffs, test_dir, global_feat_dict=global_feat_dict)
    #print "done extracting test features"
    #print
    
    # TODO make predictions on text data and write them out
    error = 0
    total = X_train.shape[0]
    print "making predictions..."
    #preds = np.argmax(X_test.dot(learned_W),axis=1)
    for index, feats in enumerate(X_train):
        prediction = predictor(feats)
        if (prediction != t_train[index]):
            print "%s: expected %d but got %d" % (train_ids[index], t_train[index], prediction)
            error += 1
    print "Correct: %d, Incorrect: %d, Total: %d, Accuracy: %f" % (total - error, error, total, (total - error) / (1.0 * total))
    print "done making predictions"
    print
    
    print "writing predictions..."
    util.write_predictions(preds, test_ids, outputfile)
    print "done!"

if __name__ == "__main__":
    main("load" in sys.argv)
    