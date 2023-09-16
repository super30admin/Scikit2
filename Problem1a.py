import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
import pickle

df = pd.read_csv('Iris.csv')
print(df.info())

# Drop Id column 
df.drop('Id',axis=1,inplace=True)
print(df.head())

#Modifying the labels to numbers
df['Species'] = df['Species'].map({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2})

# Create features and labels
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

# Splitting into train test data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

# Create RF model
clf = RandomForestClassifier()

# Fit the model on training data
clf.fit(X_train,y_train)

# Predict on the test data
y_pred = clf.predict(X_test)
report = classification_report(y_test,y_pred)
# Accuracy 
print(f'Accuracy using RF {accuracy_score(y_pred,y_test):0.3}')
print(f'Classification Report {report}')


# Using pickle, which is used for serializing and deserializing a Python object structure
pickle_file = open("rf_classifier.pkl", "wb")
pickle.dump(clf,pickle_file)
pickle_file.close()
