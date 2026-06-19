import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score 

datatrain = pd.read_csv('grypa1-train.csv')
datatest = pd.read_csv('grypa1-test.csv')
zmiana1 = ['dreszcze', 'grypa', 'katar', 'goraczka', 'bol_glowy']
zmiana2 = ['dreszcze', 'katar', 'goraczka', 'bol_glowy']
df_train = datatrain.copy()
df_test = datatest.copy()
df_train['grypa'] = df_train['grypa'].astype(str).str.strip().str.lower()
konwersja = {'tak': 1, 'nie': 0, 'sredni': 1, 'duzy': 2}
df_train[zmiana1] = df_train[zmiana1].replace(konwersja)
df_test[zmiana2] = df_test[zmiana2].replace(konwersja)

df_train['grypa'] = pd.to_numeric(df_train['grypa'], errors='coerce')
df_train = df_train.dropna(subset=['grypa'])

X = df_train.drop(columns=['grypa']).values
Y = df_train['grypa'].values
X_test = df_test.values

dt = DecisionTreeClassifier()
clr = dt.fit(X, Y)
# test data prediction (X_test)
ypred_test = clr.predict(X_test)
print("Prediction for test dataset:")
print(ypred_test)
# Training dataset prediction (X)
ypred_train = clr.predict(X)
print("\nPrediction for training dataset:")
print(ypred_train)
# accuracy on training dataset
accuracy = accuracy_score(Y, ypred_train)
print(f"\nDokladnosc na zbiorze treningowym: {accuracy}")


# [1]
# [0 1 1 1 0 1 0 1]
# 1.0
