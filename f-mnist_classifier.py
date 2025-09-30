# ========================== # PREPROCESSING # ========================== 

import pandas as pd 
from sklearn.model_selection import train_test_split 

df_train = pd.read_csv("fashion-mnist_train.csv") 
X_train = df_train.drop("label", axis=1).values 
y_train = df_train["label"].values 

X_train = X_train/255.0  # Normalizing to get values from 0 to 1 

df_test = pd.read_csv("fashion-mnist_test.csv") 
X_test = df_test.drop("label", axis=1).values 
y_test = df_test["label"].values 

X_test = X_test/255.0 

# may shorten test and train datasets to get quick results from smaller data, e.g., X_train = X_train[:1000] 


# ========================== # LOGISTIC REGRESSION # ========================== 

from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, classification_report 

log_reg = LogisticRegression(max_iter=1000, solver='saga', multi_class='multinomial') 
log_reg.fit(X_train, y_train) 
y_pred = log_reg.predict(X_test) 

print("LR Accuracy:", accuracy_score(y_test, y_pred)) 
print("\nClassification Report:\n", classification_report(y_test, y_pred)) 


# ========================== # RANDOM CLASSIFIER # ========================== 

from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score 

rf = RandomForestClassifier(n_estimators=100) 
rf.fit(X_train, y_train) 
y_rf = rf.predict(X_test) 

print("RF Accuracy:", accuracy_score(y_test, y_rf)) 


# ========================== # MLP CLASSIFIER # ========================== 

from sklearn.neural_network import MLPClassifier 

mlp = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu', solver='adam', max_iter=900, random_state=42, verbose=True)  #pyramid structure - may be hypertuned 
mlp.fit(X_train, y_train) 
y_pred_mlp = mlp.predict(X_test) 

print("MLP Accuracy:", accuracy_score(y_test, y_pred_mlp)) 
print("\nMLP Classification Report:\n", classification_report(y_test, y_pred_mlp))