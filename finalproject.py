
# coding: utf-8

# In[1]:

import numpy as np
from sklearn import svm
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn import ensemble
from sklearn.metrics import roc_curve, auc


# In[2]:

fname='mydata.txt'
data = np.loadtxt(fname,  skiprows=1)
colnum=data.shape[1]


# In[4]:

# split data into training data, validation data and test data with ratio 8:1:1
P = np.random.permutation(len(data))
split1 = len(data)*8/10
split2 = len(data)*1/10
train = data[P[:split1]]
validation = data[P[split1:split1+split2]]
test = data[P[split1+split2:]]


# In[6]:

# divid data into feature matrix and label
train_X=train[:,0:colnum-1]
train_y=train[:,colnum-1]
val_X=validation[:,0:colnum-1]
val_y=validation[:,colnum-1]
test_X=test[:,0:colnum-1]
test_y=test[:,colnum-1]


# In[34]:

def getSensitivity(y_pre,y):
    return recall_score(y, y_pre) 


# In[46]:

def svm_model(X, y,X_val):
    model = svm.SVC(C=1.0, kernel='linear', class_weight='balanced')
    model.fit(X, y)
    y_te=model.predict(X_val)
    return y_te, model


# In[36]:

def svm_grid_search(X,y,X_val,y_val):
    kernels=['linear','rbf','poly','sigmoid']
    max_tpr=0
    best=''
    for k in kernels:
        svm_predict =svm_model(X, y,k, X_val)
        tpr=getSensitivity(svm_predict,y_val)
        if tpr>max_tpr:
            max_tpr=tpr
            best=k
    print max_tpr
    print k
    return max_tpr


# In[47]:

#train svm model
model_result=svm_model(train_X, train_y,val_X)
svm_pre=model_result[0]
svm_model=model_result[1]
performance_svm=getSensitivity(svm_pre,val_y)


# In[79]:

#show trained weights
weight=svm_model.coef_ 
weights=weight.flatten()
np.set_printoptions(formatter={'float_kind':'{:f}'.format})
weights


# In[76]:

#logistic regression
def lr_model(X, y,X_val):
    model = LogisticRegression(penalty='l2',class_weight='balanced', solver='liblinear', max_iter=100)
    model.fit(X, y)
    y_te=model.predict(X_val)
    return y_te


# In[77]:

#train logistic regression
logistic_pre=lr_model(train_X, train_y,val_X)
performance_log=getSensitivity(logistic_pre,val_y)


# In[14]:

#gradientboosting tree
def GBTree(X,y):
    from sklearn.grid_search import GridSearchCV 
    param_grid = {'learning_rate': [0.1, 0.05, 0.02, 0.01], 
              'max_depth': [3,4,6], 
              'min_samples_leaf': [3, 5, 9, 17], 
              'max_features': [1.0, 0.3, 0.1]} 
    clf = ensemble.GradientBoostingClassifier(n_estimators=100)
    gs = GridSearchCV(clf, param_grid).fit(X, y) 
    return gs.best_params_
    pass
def tree_model(X, y,X_val,param):
    params = {'n_estimators': 500, 'max_depth': param['max_depth'], 'min_samples_leaf': param['min_samples_leaf'],
          'learning_rate':param['learning_rate'], 'max_features':param['max_features'] }
    clf = ensemble.GradientBoostingClassifier(**params)
    clf.fit(X, y)
    y_pre=clf.predict(X_val)
    return y_pre


# In[19]:

#grid search of GBTree
tree_param=GBTree(train_X,train_y)
tree_param


# In[22]:

# train GBTree
tree_pre=tree_model(train_X, train_y,val_X,tree_param)
performance_tree=getSensitivity(tree_pre,val_y)

