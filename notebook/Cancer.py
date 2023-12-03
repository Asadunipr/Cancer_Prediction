import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

Data = pd.read_csv("C:\Users\Akhan\Desktop\Learning_2023\MACHINE LEARNING\Cancer_Prediction\notebook\data\data.csv")
Data.head()

X = Data.drop(["id", "diagnosis", "Unnamed: 32"], axis = 1)

X.head()

Data.info()

Data.isna().sum()

y = Data["diagnosis"]

y.value_counts()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state= 42)

from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()

X_train = scalar.fit_transform(X_train)

X_test =scalar.fit_transform(X_test)

X_train

from sklearn.linear_model import LogisticRegression


model_L = LogisticRegression()

model_L.fit(X_train, y_train)

from sklearn.metrics import classification_report, accuracy_score

model_L_preds = model_L.predict(X_test)

model_L_acc=accuracy_score(y_test, model_L_preds)
model_L_acc

classification_report(y_test, model_L_preds)

from sklearn.metrics import confusion_matrix

cm_L =confusion_matrix(y_test, model_L_preds)
cm_L

from sklearn.tree import DecisionTreeClassifier

model_D = DecisionTreeClassifier(random_state =42)

model_D.fit(X_train, y_train)

model_D_preds =model_D.predict(X_test)
model_D_preds

model_D_acc = accuracy_score(y_test, model_D_preds)
model_D_acc

classification_report(y_test, model_D_preds)

confusion_matrix(y_test, model_D_preds)

from sklearn.ensemble import RandomForestClassifier

model_R =RandomForestClassifier(n_estimators=100, random_state= 42)

model_R.fit(X_train, y_train)

model_R_preds =model_R.predict(X_test)
model_R_preds

model_R_acc = accuracy_score(y_test, model_R_preds)
model_R_acc

classification_report(y_test, model_R_preds)

confusion_matrix(y_test, model_R_preds)

from sklearn.naive_bayes import GaussianNB

model_G = GaussianNB()

model_G.fit(X_train, y_train)

model_G_preds = model_G.predict(X_test)
model_G_preds

model_G_acc = accuracy_score(y_test, model_G_preds)
model_G_acc

classification_report(y_test, model_G_preds)

confusion_matrix(y_test, model_G_preds)

from sklearn.neighbors import KNeighborsClassifier

model_KN =KNeighborsClassifier(n_neighbors = 2)

model_KN.fit(X_train, y_train)

model_KN_preds = model_KN.predict(X_test)
model_KN_preds

model_KN_acc = accuracy_score(y_test, model_KN_preds)
model_KN_acc

classification_report(y_test, model_KN_preds)

confusion_matrix(y_test, model_KN_preds)

from sklearn.neural_network import MLPClassifier

model_neural = MLPClassifier()

model_neural.fit(X_train, y_train)

model_neural_preds =model_neural.predict(X_test)
model_neural_preds

model_neural_acc = accuracy_score(y_test, model_neural_preds)
model_neural_acc

classification_report(y_test, model_neural_preds)

confusion_matrix(y_test, model_neural_preds)

from sklearn.svm import SVC

model_svc = SVC()

model_svc.fit(X_train, y_train)

model_svc_preds = model_svc.predict(X_test)
model_svc_preds

model_svc_acc = accuracy_score(y_test, model_svc_preds)
model_svc_acc


classification_report(y_test, model_svc_preds)

confusion_matrix(y_test, model_svc_preds)

from xgboost import XGBClassifier

model_XGB = XGBClassifier()

#local_mapping ={"B" : 0, "M" : 1}




# Encode the label (loan_status) column
    #label_mapping = {'COLLECTION': 0, 'PAIDOFF': 1, 'COLLECTION_PAIDOFF': 2}
   # df['loan_status'] = df['loan_status'].replace(label_mapping)

#Data.y.replace({'M' : '1','B': '0'},inplace=True)

#model_XGB.fit(X_train, y_train)




