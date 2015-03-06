import os
import sys
import numpy as np
import random
import pickle
from datetime import datetime
from scipy import stats
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
import util
from classifier import extract_feats, sk_logistic
import extractors
from extractors import ffs
from randomforest_classifier import sk_random_forest

## The following function does the feature extraction, learning, and prediction
def main(load = False, test=False, both=False):
    train_dir = "train"
    test_dir = "test"
    outputfile = "treepredictions.csv"  # feel free to change this or take it as an argument
    
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
    else:
        print "Loading previous features"
        with open("X_train", "r")           as out: X_train =           pickle.load(out)
        with open("global_feat_dict", "r")  as out: global_feat_dict =  pickle.load(out)
        with open("t_train", "r")           as out: t_train =           pickle.load(out)
        with open("train_ids", "r")         as out: train_ids =         pickle.load(out)
        print "Done loading"
        print

    # if we're verifying things, save some test data
    if not test:
        print "Getting holdout data..."
        Xs, ts, ids = (X_train, t_train, train_ids)
        n = Xs.shape[0]
        train_pct = 0.8

        X_train = Xs[-int(n*train_pct):]
        t_train = ts[-int(n*train_pct):]
        train_ids = ids[-int(n*train_pct):]
        
        X_holdout = Xs[:-int(n*train_pct)]
        t_holdout = ts[:-int(n*train_pct)]
        holdout_ids = ids[:-int(n*train_pct)]
        print
    # TODO train here, and learn your classification parameters
    print "learning..."
    num_trees = 100
    forest = RandomForestClassifier(n_estimators = num_trees)
    forest = forest.fit(X_train.todense(), t_train)
    # Random forest predictor
    forest_predictor, _ = sk_random_forest(X_train.toarray(), t_train, num_trees = num_trees)
    # logistic regression predictor
    # log_predictor, _ = sk_logistic(X_train, t_train)
    print "done learning"
    print
    
    # get rid of training data and load test data
    # del X_train
    # del t_train
    # del train_ids
    
    
    # if you want to write predictions for test data
    if test:
        # if you didn't save both sets of features, extract
        if not both:
            print "extracting test features..."
            X_test,_,t_ignore,test_ids = extract_feats(ffs, test_dir, global_feat_dict=global_feat_dict)
            print "done extracting test features"
            print
            print "Saving test features"
            with open("X_test", "w") as out:
                pickle.dump(X_test, out)
            with open("test_ids", "w") as out:
                pickle.dump(test_ids, out)
            print "Done saving"
            print
        else:
            print "Loading previous test features"
            with open("X_test", "r")           as out: X_test =           pickle.load(out)
            with open("test_ids", "r")         as out: test_ids =         pickle.load(out)
            print "Done loading"
            print
        # TODO make predictions here
        print "making predictions..."
        preds = forest.predict(X_test.toarray())
        print "done making predictions"

        print "writing predictions..."
        util.write_predictions(preds, test_ids, outputfile)
        print "done!"
    else:
        error = 0
        total = X_holdout.shape[0]
        print "making predictions..."
        #preds = np.argmax(X_test.dot(learned_W),axis=1)
        #preds = logreg.predict(X_test)
        random.seed(datetime.now())
        for index, feats in enumerate(X_holdout.toarray()):
            pred_forest = forest_predictor(feats)
            # pred_logistic = log_predictor(feats)
            # #if they agree, or disagree and both predict malware
            # if pred_forest == pred_logistic or (pred_forest != 8 and pred_logistic != 8):
            #     prediction = pred_forest
            # else:
            #     # grab the non-"None" label
            #     other = pred_forest if pred_forest != 8 else pred_logistic
            #     # flip a coin
            #     if random.random() < 0.39:
            #         prediction = 8
            #     else:
            #         prediction = other
            prediction = pred_forest

            if (prediction != t_holdout[index]):
                print "%s: expected %d but got %d" % (holdout_ids[index], t_holdout[index], prediction)
                error += 1
        print "Correct: %d, Incorrect: %d, Total: %d, Accuracy: %f" % (total - error, error, total, (total - error) / (1.0 * total))
        print "done making predictions"
        print
    print



if __name__ == "__main__":
    main("load" in sys.argv, "test" in sys.argv, "both" in sys.argv)