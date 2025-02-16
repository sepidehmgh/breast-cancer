# breast-cancer 
from sklearn.datasets import load_breast_cancer 
bc=load_breast_cancer()
print(bc.DESCR)

.. _breast_cancer_dataset:

Breast cancer wisconsin (diagnostic) dataset
--------------------------------------------

**Data Set Characteristics:**

:Number of Instances: 569

:Number of Attributes: 30 numeric, predictive attributes and the class

:Attribute Information:
    - radius (mean of distances from center to points on the perimeter)
    - texture (standard deviation of gray-scale values)
    - perimeter
    - area
    - smoothness (local variation in radius lengths)
    - compactness (perimeter^2 / area - 1.0)
    - concavity (severity of concave portions of the contour)
    - concave points (number of concave portions of the contour)
    - symmetry
    - fractal dimension ("coastline approximation" - 1)

    The mean, standard error, and "worst" or largest (mean of the three
    worst/largest values) of these features were computed for each image,
...
  - W.H. Wolberg, W.N. Street, and O.L. Mangasarian. Machine learning techniques
    to diagnose breast cancer from fine-needle aspirates. Cancer Letters 77 (1994)
    163-171.

bc.target[0]
bc.target.shape
bc.data[500]
bc.data.shape

#train/test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(bc.data,bc.target,test_size=0.2)

x_test.shape,x_train.shape

x_train[0]

#normalise
from sklearn.preprocessing import MinMaxScaler
scalar=MinMaxScaler(feature_range=(0,1))
x_train=scalar.fit_transform(x_train)
x_test=scalar.fit_transform(x_test)

x_train.shape
x_train[0]

from sklearn.metrics import accuracy_score, recall_score,confusion_matrix,precision_score

def calculate_metrics(y_train,y_test,y_pred_train,y_pred_test):

    conf_train=confusion_matrix(y_train,y_pred_train)
    conf_test=confusion_matrix(y_test,y_pred_test)

    acc_train=accuracy_score(y_train,y_pred_train)
    acc_test=accuracy_score(y_true=y_test,y_pred=y_pred_test)

    rec_train=recall_score(y_train,y_pred_train)
    rec_test=recall_score(y_test,y_pred_test)

    prec_train=precision_score(y_train,y_pred_train)
    prec_test=precision_score(y_test,y_pred_test)
    
    print(
      f"confusion for training data: {conf_train}\t\t"
      f"confusion for test data: {conf_test}\n\n"
      f"Accuracy for training data: {acc_train}\t\t"
      f"Accuracy for test data: {acc_test}\n\n"
      f"Recall for training data: {rec_train}\t\t"
      f"Recall for test data: {rec_test}\n\n"
      f"Precision for training data: {prec_train}\t\t"
      f"Precision for test data: {prec_test}\n\n")

    return  conf_train,conf_test,acc_train,acc_test,rec_train,rec_test,prec_train,prec_test

#naive bayes 
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(x_train,y_train)

y_pred_train=gnb.predict(x_train)
y_pred_test=gnb.predict(x_test)

conf_train_gnb, conf_test_gnb, acc_train_gnb,acc_test_gnb, rec_train_gnb,rec_test_gnb,prec_train_gnb,prec_test_gnb=calculate_metrics(y_train,y_test,y_pred_train,y_pred_test)


#knn
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=8,algorithm='kd_tree',leaf_size=28)
knn.fit(x_train,y_train)
y_pred_train=gnb.predict(x_train)
y_pred_test=gnb.predict(x_test)

y_pred_train=knn.predict(x_train)
y_pred_test=knn.predict(x_test)
conf_train_knn, conf_test_knn, acc_train_knn,acc_test_knn, rec_train_knn,rec_test_knn,prec_train_knn,prec_test_knn=calculate_metrics(y_train,y_test,y_pred_train,y_pred_test)



#DT
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(max_depth=128,min_samples_leaf=2,min_samples_split=4,criterion='entropy')
dt.fit(x_train,y_train)
y_pred_train=dt.predict(x_train)
y_pred_test=dt.predict(x_test)
conf_train_dt, conf_test_dt, acc_train_dt,acc_test_dt, rec_train_dt,rec_test_dt,prec_train_dt,prec_test_dt=calculate_metrics(y_train,y_test,y_pred_train,y_pred_test)


#RF
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=500, max_depth=64, min_samples_split=8)
rf.fit(x_train,y_train)
y_pred_train=rf.predict(x_train)
y_pred_test=rf.predict(x_test)
conf_train_rf, conf_test_rf, acc_train_rf,acc_test_rf, rec_train_rf,rec_test_rf,prec_train_rf,prec_test_rf=calculate_metrics(y_train,y_test,y_pred_train,y_pred_test)


#SVM
from sklearn.svm import SVC
svm=SVC(kernel='poly')
svm.fit(x_train,y_train)
y_pred_train=svm.predict(x_train)
y_pred_test=svm.predict(x_test)
conf_train_svm, conf_test_svm, acc_train_svm,acc_test_svm, rec_train_svm,rec_test_svm,prec_train_svm,prec_test_svm=calculate_metrics(y_train,y_test,y_pred_train,y_pred_test)

#Lr
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
y_pred_train=lr.predict(x_train)
y_pred_test=lr.predict(x_test)
conf_train_lr, conf_test_lr, acc_train_lr,acc_test_lr, rec_train_lr,rec_test_lr,prec_train_lr,prec_test_lr=calculate_metrics(y_train,y_test,y_pred_train,y_pred_test)

#ANN
from sklearn.neural_network import MLPClassifier
ann=MLPClassifier(hidden_layer_sizes=256,activation='relu',solver='adam',batch_size=64)
ann.fit(x_train,y_train)
y_pred_train=ann.predict(x_train)
y_pred_test=ann.predict(x_test)
conf_train_ann, conf_test_ann, acc_train_ann,acc_test_ann, rec_train_ann,rec_test_ann,prec_train_ann,prec_test_ann=calculate_metrics(y_train,y_test,y_pred_train,y_pred_test)

#comparison

import matplotlib.pyplot as plt
acc_train=[acc_train_gnb,acc_train_knn,acc_train_dt,acc_train_rf,acc_train_svm,acc_train_lr,acc_train_ann]
title=["gnb",'knn','dt','rf','svm','lr','ann']
colors=['black','red','yellow','orange','green','blue','pink']
plt.bar(title,acc_train,color=colors)
plt.grid()
plt.show()

acc_test=[acc_test_gnb,acc_test_knn,acc_test_dt,acc_test_rf,acc_test_svm,acc_test_lr,acc_test_ann]
title=["gnb",'knn','dt','rf','svm','lr','ann']
colors=['black','red','yellow','orange','green','blue','pink']
plt.bar(title,acc_test,color=colors)
plt.grid()
plt.show()

