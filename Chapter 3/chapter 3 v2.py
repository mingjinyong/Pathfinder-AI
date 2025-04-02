from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler



#Fetch the MNIST dataset
mnist = fetch_openml( 'mnist_784', version=1, parser="auto", as_frame=False)

X,y = mnist["data"], mnist["target"]

#Let's look an an instance's feature vector
some_digit = X[0] #This is a '5'

#y is a bunch of text, not numbers. Let's convert to a number - better for computer
y = y.astype(np.uint8)

#Divide the 70,000 rows into the train set and test set
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

#print(len(X_train), len(X_test))  #Just to check we did it correctly

#Create two new objects that only have the images of 5
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
print(sgd_clf.predict([some_digit]))

'''
skfolds = StratifiedKFold(n_splits=3, random_state=42)
for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred== y_test_fold)
    print(n_correct / len(y_pred)) #Supposed to print 0.9502, 0.96565 and 0.96495
'''

#print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

print("Precision:",precision_score(y_train_5, y_train_pred))
print("Recall:",recall_score(y_train_5, y_train_pred))

#Compute the F1 score
#print("F1 score:",f1_score(y_train_5, y_train_pred))

'''
y_scores = sgd_clf.decision_function([some_digit])
print(y_scores)

threshold = 0

y_some_digit_pred = (y_scores > threshold)
print("Digit prediction 0 threshold:",y_some_digit_pred)
'''
'''
threshold = 200000
y_some_digit_pred = (y_scores > threshold)
print("Digit prediction 200000 threshold:",y_some_digit_pred)
'''

'''
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
plt.xlabel("Treshold")
plt.legend(loc="center left")
plt.ylim([0,1])
plt.show()
'''

''' 
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
plt.plot(fpr,tpr,linewidth=2,label=None)
plt.plot([0,1],[0,1], 'k--') #Plots 45 degree dashed line as black
plt.axis([0,1,0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
'''

'''
sgd_clf.fit(X_train,y_train)  #y_train not y_train_5
print(sgd_clf.predict([some_digit])) #The model's prediction for our original 5 - row 1
'''

''' 
some_digit_scores = sgd_clf.decision_function([some_digit])
print(some_digit_scores)
'''

'''
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()
'''

'''
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums #Normalize the confusion matrix
np.fill_diagonal(norm_conf_mx,0)  #Fill the diagonals with zeros to keep only the errors
plt.matshow(norm_conf_mx, 0)
plt.show()
'''