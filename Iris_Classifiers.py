import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("Iris.csv")
print(df.head())
print(df.info())
print(df.isnull().sum())
print(df["Species"].value_counts())

print(df.describe())
df.drop("Id",axis=1,inplace=True)
print(df.info())

sns.pairplot(df)
plt.show()

sns.scatterplot(x=df["SepalLengthCm"],y=df["SepalWidthCm"],hue=df["Species"])
plt.show()
sns.scatterplot(x=df["SepalLengthCm"],y=df["PetalWidthCm"],hue=df["Species"])
plt.show()

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

df["Species"] = label_encoder.fit_transform(df["Species"])
print(df.head())
print(df.tail())

print(df["Species"].value_counts())

X = df.drop("Species",axis=1)
y = df["Species"]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)
# scaler gerekmiyor olsa bile eğer gaussion naive bayes kullanıyor isek standardscaler yapmamız daha uygun olur 

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train_scaled,y_train)
y_pred = gnb.predict(X_test_scaled)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print("Accuracy Score: ",accuracy_score(y_test,y_pred))
print("Confusion_matrix: \n",confusion_matrix(y_test,y_pred))
print("Classification_report: \n",classification_report(y_test,y_pred))

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train_scaled,y_train)

y_pred = logreg.predict(X_test_scaled)
print("Logistic Regression Results")
print("Accuracy Score: ",accuracy_score(y_test,y_pred))
print("Classification_report: \n",classification_report(y_test,y_pred))
print("Confusion_matrix: \n",confusion_matrix(y_test,y_pred))
print("\n")

from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train_scaled, y_train)

y_pred = svc.predict(X_test_scaled)

print("SVC Results")
print("Accuracy Score: ",accuracy_score(y_test,y_pred))
print("Classification_report: \n",classification_report(y_test,y_pred))
print("Confusion_matrix: \n",confusion_matrix(y_test,y_pred))
print("\n")


model = LogisticRegression()
c_values = [1000,100,10,1,0.1,0.01] 

# {'penalty': ['l1', 'l2', 'elasticnet'], 'C': [100, 10, 1, 0.1, 0.01], 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga', 'newton-cholesky']} 
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

cv = StratifiedKFold()

params = [
    {"penalty": ["l2"], "solver": ["newton-cg", "lbfgs", "sag", "saga", "newton-cholesky"], "C": c_values},
    {"penalty": ["l1"], "solver": ["liblinear", "saga"], "C": c_values},
    {"penalty": ["elasticnet"], "solver": ["saga"], "C": c_values, "l1_ratio": [0.5]} 
]

randomcv = RandomizedSearchCV(estimator=model, param_distributions=params, cv=cv, scoring="accuracy")
randomcv.fit(X_train_scaled, y_train)
y_pred = randomcv.predict(X_test_scaled)

print("Logistic Regression Hyperparameter Tuning Results")
print("Accuracy Score: ",accuracy_score(y_test,y_pred))
print("Classificaion Report: \n",classification_report(y_test,y_pred))
print("Confusion Matrix: \n",confusion_matrix(y_test,y_pred))

print("randombest_params",randomcv.best_params_)
print("randombest_score:",randomcv.best_score_)


from sklearn.model_selection import GridSearchCV

params = [
    {"penalty": ["l2"], "solver": ["newton-cg", "lbfgs", "sag", "saga", "newton-cholesky"], "C": c_values},
    {"penalty": ["l1"], "solver": ["liblinear", "saga"], "C": c_values},
    {"penalty": ["elasticnet"], "solver": ["saga"], "C": c_values, "l1_ratio": [0.5]} 
]

grid = GridSearchCV(estimator=model, param_grid=params, cv=cv, scoring="accuracy", n_jobs=-1)

grid.fit(X_train_scaled,y_train)

y_pred = grid.predict(X_test_scaled)

print("Accuracy Score: ",accuracy_score(y_test,y_pred))
print("Classification_report: \n",classification_report(y_test,y_pred))
print("Confusion_matrix: \n",confusion_matrix(y_test,y_pred))

print(grid.best_params_)
print(grid.best_score_)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Classes")
plt.ylabel("Real Classes")
plt.title("Log_Reg - Confusion Matrix Heatmap")
plt.show()

from sklearn.svm import SVC

svc_params = {
    "C": [0.1, 1, 10, 100, 1000],
    "gamma": [1, 0.1, 0.01, 0.001],
    "kernel": ["rbf", "linear"]
}

svc_model = SVC()

grid_svc = GridSearchCV(estimator=svc_model, param_grid=svc_params, cv=cv, scoring="accuracy", n_jobs=-1)

grid_svc.fit(X_train_scaled, y_train)

y_pred_svc = grid_svc.predict(X_test_scaled)

print("SVC Hyperparameter Tuning Sonuçları")
print("-----------------------------------")
print("En İyi Parametreler:", grid_svc.best_params_)
print("En İyi Çapraz Doğrulama Skoru (Eğitim):", grid_svc.best_score_)
print("\nTest Seti Accuracy Score:", accuracy_score(y_test, y_pred_svc))
print("\nClassification Report:\n", classification_report(y_test, y_pred_svc))


cm_svc = confusion_matrix(y_test, y_pred_svc)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_svc, annot=True, fmt="d", cmap="Greens") 
plt.xlabel("Predicted Classes")
plt.ylabel("Real Classes")
plt.title("SVC - Confusion Matrix Heatmap")
plt.show()
