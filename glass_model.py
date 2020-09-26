import numpy as np
import pandas as pd

import pickle

df = pd.read_csv('glass.csv')

# X
X = np.array(df[['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe']])

# Y
key = {'BWNF':0,'BWF':1,'Headlamps':2,'VWF':3,'Containers':4,'Tableware':5}
Y = df['Type'].map(key)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
model_logis = LogisticRegression()
model_logis.fit(X_train_std,y_train)

from sklearn.tree import DecisionTreeClassifier
model_dtree = DecisionTreeClassifier(criterion='gini',max_depth=4,presort=True)
model_dtree.fit(X_train_std,y_train)

from sklearn.ensemble import RandomForestClassifier
model_forrest = RandomForestClassifier(max_depth=2)
model_forrest.fit(X_train_std,y_train)

from sklearn.svm import LinearSVC
model_svc = LinearSVC(C=10)
model_svc.fit(X_train_std,y_train)

from sklearn.neighbors import KNeighborsClassifier
model_knc = KNeighborsClassifier(n_neighbors=3)
model_knc.fit(X_train_std,y_train)

pickle.dump(model_logis,open('logis_model.pkl','wb'))
pickle.dump(model_dtree,open('dtree_model.pkl','wb'))
pickle.dump(model_forrest,open('forrest_model.pkl','wb'))
pickle.dump(model_svc,open('svc_model.pkl','wb'))
pickle.dump(model_knc,open('knc_model.pkl','wb'))