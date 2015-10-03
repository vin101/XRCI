'''from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt

def main():
    #create the training & test sets, skipping the header row with [1:]
    dataset = genfromtxt(open('/home/arbaaz/projects/kaggle/train.csv','r'), delimiter=',', dtype='f8')[1:]    
    target = [x[0] for x in dataset]
    train = [x[1:] for x in dataset]
    test = genfromtxt(open('/home/arbaaz/projects/kaggle/test.csv','r'), delimiter=',', dtype='f8')[1:]
    
    #create and train the random forest
    #multi-core CPUs can use: 
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    #rf = RandomForestClassifier(n_estimators=100)
    rf.fit(train, target)

    savetxt('/home/arbaaz/projects/kaggle/submission2.csv', rf.predict(test), delimiter=',', fmt='%f')

if __name__=="__main__":
    main()'''
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation
import logloss
import numpy as np

def main():
    #read in  data, parse into training and target sets
    dataset = np.genfromtxt(open('train.csv','r'), delimiter=',', dtype='f8')[1:]    
    target = np.array([x[0] for x in dataset])
    train = np.array([x[1:] for x in dataset])
    #print target

    #In this case we'll use a random forest, but this could be any classifier
    #rf = ExtraTreesClassifier(n_estimators=100, n_jobs=-1)
    #cfr = AdaBoostClassifier(n_estimators=80, learning_rate=1.0)
    rf = AdaBoostClassifier(
    learning_rate=1,
    n_estimators=80,
    algorithm="SAMME.R")
    #rf = GradientBoostingClassifier(loss = 'deviance', learning_rate = 0.12, n_estimators = 400, max_depth = 3, verbose = 1)


    #Simple K-Fold cross validation. 5 folds.
    #(Note: in older scikit-learn versions the "n_folds" argument is named "k".)
    cv = cross_validation.KFold(len(train), n_folds=10, indices=False)




    #iterate through the training and test cross validation segments and
    #run the classifier on each one, aggregating the results into a list
    results = []
    for traincv, testcv in cv:
        probas = rf.fit(train[traincv], target[traincv]).predict_proba(train[testcv])
        results.append( logloss.llfun(target[testcv], [x[1] for x in probas]) )

    #print out the mean of the cross-validated results
    print "Results: " + str( np.array(results).mean() )

if __name__=="__main__":
    main()
