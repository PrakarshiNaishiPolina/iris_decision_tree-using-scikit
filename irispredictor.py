
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

iris=load_iris()
print(iris.keys()) # to see whats in the dataset

# convert to a dataframe 
df=pd.DataFrame(iris.data,columns=iris.feature_names)

# the data is taken from iris.data and column names are taken using iris.feature_names

df['species']=iris.target

# a new column named species is added to the df, the data for this column comes from iris.target

print(df.head()) # preview data

# Split data into training and testing sets

X=iris.data # stores the input data
y=iris.target # output or stores the labels with respect to X.

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5,random_state=42)

# Train a model

model=DecisionTreeClassifier()
model.fit(X_train,y_train)

# Make predictions

y_pred=model.predict(X_test) #  use the model to predict the labels for the test dataset.

# result is stored in y_pred, which is an array of predicted labels.

# Evaluate the model
accuracy=accuracy_score(y_test,y_pred) # calcualtes the accuracy of the model by comparing the actual labels(y_test) with predicted labels (y_pred)

print(f"Accuracy: {accuracy*100:.2f}%") # predicts the accuracy as percentage with 2 decimal points.

# Test with new data

new_data=[[5.1,3.5,1.4,0.2]]

#  Example Input

prediction=model.predict(new_data)

# the model is used to predict the species for the new data

print("Predicted species:",iris.target_names[prediction[0]])

# iris.target_names is an array that contains the name of the species corresponding to the numeric target labels in the iris dataset

# prediction is an array containing the output of the model's prediction. For example:
# If model.predict(new_data) returns [0], the model predicts the species as 0 (Setosa).

# prediction is an array, even if there is only one prediction. To extract the actual numeric label (e.g., 0), we access the first element of the array using prediction[0].