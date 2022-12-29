


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
dataset = pd.read_csv(r"diabetes (1).csv")
dataset.head()
dataset.shape
dataset.describe()
sns.countplot(x = 'Outcome', data = dataset)

corr_mat = dataset.corr()
sns.heatmap(corr_mat, annot = True)
plt.show()

dataset.isna().sum()

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
X[0]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
x_train.shape

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
x_train[0]

#support vector machine
from sklearn.svm import SVC
SVC_model=SVC()
SVC_model.fit(x_train,y_train)

SVC_pred=SVC_model.predict(x_test)

from sklearn import metrics
print("Accuracy score= ",format(metrics.accuracy_score(y_test,SVC_pred)))

#random forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(x_train,y_train)

rfc_train=rfc.predict(x_train)
from sklearn import metrics

print("Accuracy score= ", format(metrics.accuracy_score(y_train,rfc_train)))


from sklearn import metrics
predictions=rfc.predict(x_test)
print("Accuracy Score= ", format(metrics.accuracy_score(y_test,predictions)))

#knn
from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier(n_neighbors=25, metric="minkowski")
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(x_train,y_train)

from sklearn import metrics
predictions = dtree.predict(x_test)
print("Accuracy_score= ", format(metrics.accuracy_score(y_test,predictions)))

#gradient booster
from sklearn.ensemble import GradientBoostingClassifier
xgb_model = GradientBoostingClassifier(n_estimators = 300, max_features = 2, max_depth = 9,random_state = 0)
xgb_model.fit(x_train,y_train)

from sklearn import metrics
xgb_pred=xgb_model.predict(x_test)
print("Accuracy_score= ", format(metrics.accuracy_score(y_test,xgb_pred)))







