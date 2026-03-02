import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter

df = pd.read_csv('ml_classification_data.csv')

#Need to create arrays
X = df.drop('label',axis=1).values
y = df['label'].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

#Need to scale data as KNN is distance based
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def euclidean_distance(p,q):
    return np.sqrt(np.sum((p-q)**2))

class KNN:
    def __init__(self,k=3):
        self.k = k

    def fit(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self,X_test):
        predictions = []

        for test_point in X_test:
            distances = []

#Use enumerate to get both index and vaue point
            for i,train_point in enumerate(self.X_train):
                distance = euclidean_distance(train_point,test_point)
                distances.append([distance,self.y_train[i]])

            #sort by distance
            distances.sort(key=lambda x: x[0])

            #take first k labels
            k_nearest = distances[:self.k]
            labels = [label for _,label in k_nearest]

            #majority vote
            most_common = Counter(labels).most_common(1)
            predictions.append(most_common[0][0])    

        return np.array(predictions)


model_scratch = KNN(k=3)
model_scratch.fit(X_train,y_train)
y_pred_scratch = model_scratch.predict(X_test)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

plt.scatter(range(len(y_test)), y_test,color='black')
plt.scatter(range(len(y_pred_scratch)), y_pred_scratch,color='red')
plt.scatter(range(len(y_pred)),y_pred,color='green')
plt.legend()
plt.show()
