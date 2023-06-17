### importing libraries 
import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.tree import DecisionTreeClassifier, plot_tree


import warnings
warnings.filterwarnings("ignore")



### Loading the data
data = pd.read_csv('spam.csv', encoding='latin')
data.shape


print(data.head())

print(data.tail())

print(data.info())

print(data.describe())

### Data Cleaning 

# Drop unnecessary columns
data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1, inplace=True)
print(data.head())

# Update the colum names 
data.rename(columns={"v1":"type", "v2":"message"}, inplace=True)
print(data.head())


# Change the datatype of variable type to be categorical
data["type"] = data["type"].astype("category")
print(data.info())


###   Feature Engineering and Univariate Analysis

# From message, extract the length and create a new feature that contains the length of each message
data['message_length'] = data['message'].apply(len)
print(data.head())


#create an instance of label encoder
encoder = LabelEncoder()

#label encode the variable type [0, 1]
data["type"] = encoder.fit_transform(data["type"])  
print(data.head())

### Creating Training and Test set

# Vectorize the variable message that will be used as x
count = CountVectorizer()

x = count.fit_transform(data['message'])
y = data["type"]

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.20, random_state=52)

# Shape of train data
print((x_train.shape), (y_train.shape))

# Shape of test data
print((x_test.shape), (y_test.shape))


### Creating and Training the model

# Creating an instance of Decision Tree Classifier model
model = DecisionTreeClassifier(random_state=99)

# Fitting the model
model.fit(x_train, y_train)

# Make predictions on the test set
prediction = model.predict(x_test)
print(prediction)

### Evaluating the model

print("ACCURACY SCORE : {}". format(accuracy_score(y_test, prediction)))
print("PRECISION SCORE : {}". format(precision_score(y_test, prediction)))


##  Visualize the Tree

#create a new model with short depth to be visualized
decision_tree = DecisionTreeClassifier(max_depth=3, random_state=99)
decision_tree.fit(x_train, y_train)

#display the model tree
plt.figure("EMAIL SPAM CLASSIFIER DECISION TREE", figsize=[16, 6])
plot_tree(decision_tree, fontsize=10, filled=True)
plt.show()