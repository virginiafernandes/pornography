import pickle
import numpy as np
import os

from sklearn import svm
#from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report


# Directory containing pickles.
pkl_dir = 'tensors/'

# Train and test folds.
train_fold_file = 'DatabasePorn/training/fold0123_video.txt'
test_fold_file = 'DatabasePorn/test/fold4_video.txt'

train_list = [l.strip('\n') for l in open(train_fold_file).readlines()]
test_list  = [l.strip('\n') for l in open(test_fold_file).readlines()]

[x for x in train_list if x]
[x for x in test_list if x]

#print(train_list)
#print(test_list)

#tensor_fromvNonPorn12.avihog8.pkl

# Listing files in pkl_dir.
#files = [f for f in os.listdir(pkl_dir) if os.path.isfile(os.path.join(pkl_dir, f))]

# Initiating lists.
train_feats  = list()
train_labels = list()

test_feats  = list()
test_labels = list()

# Iterating over training files.
for i in range(len(train_list)):
    f = train_list[i].replace('\r\r', '')
    #print(f)
    
    try:
        
        # Loading pickle.
        pkl = pickle.load(open(pkl_dir + 'tensor_from' + f + '.avihog8.pkl', 'rb'))
        pkl = pkl['tensor_series'].ravel()
        # Appending feature matrix.
        train_feats.append(pkl)
        #print(pkl)
        
        # Loading label.
        lab = 1 # Initiates as porn.
        # If 'NonPorn' in file name, switch to lab = 0 (non porn).
        if 'NonPorn' in f:
            lab = 0
    
        # Appending label vector.
        train_labels.append(lab)
        
    except:
        continue

# Iterating over test files.
for i in range(len(test_list)):
    
    f = test_list[i].replace('\r\r', '')
    #print(f)
    
    try:
        
        # Loading pickle.
        pkl = pickle.load(open(pkl_dir + 'tensor_from' + f + '.avihog8.pkl', 'rb'))
        pkl = pkl['tensor_series'].ravel()
        # Appending feature matrix.
        test_feats.append(pkl)
        #print(pkl)
        
        # Loading label.
        lab = 1 # Initiates as porn.
        # If 'NonPorn' in file name, switch to lab = 0 (non porn).
        if 'NonPorn' in f:
            lab = 0
    
        # Appending label vector.
        test_labels.append(lab)

    except:
        continue

train_feats = np.asarray(train_feats)
train_labels = np.asarray(train_labels)

test_feats = np.asarray(test_feats)
test_labels = np.asarray(test_labels)
        
# Initiating SVM.
clf = svm.SVC()

# Fitting SVM to training data.
clf.fit(train_feats, train_labels)

# Generating predictions on test set.
preds = clf.predict(test_feats)

report = classification_report(test_labels, preds)

confusion_matrix = confusion_matrix(test_labels, preds)
accuracy = accuracy_score(test_labels, preds)
precision = precision_score(test_labels, preds)
recall = recall_score(test_labels, preds)

print('Confusion Matrix', confusion_matrix)
print('Accuracy', accuracy)
print('Precision', precision)
print('Recall', recall)

# Performing Crossvalidation.
#scores = cross_val_score(clf, feats, labels, cv = 5, scoring = 'accuracy')
#scores
