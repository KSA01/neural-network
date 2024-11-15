from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

# Get the iris training data
iris_data = []
with open("iris.data", "r") as file:
    for line in file:
        if len(line) == 0 or line[0] == '\n': continue
        if line[-1] == '\n': line = line[:-1]
        line = line.strip().split(",")
        features = [float(value) for value in line[:-1]]
        target = line[-1]
        iris_data.append((features, target))

# Separate features and target labels
X_iris_train = np.array([data[0] for data in iris_data])
Y_iris_train = np.array([data[1] for data in iris_data])
Y_iris_train = label_encoder.fit_transform(Y_iris_train) # Transform from strings to ints to avoid errors

# Get the iris test data
iris_test_data = []
with open("iris_test.data", "r") as file:
    for line in file:
        if len(line) == 0 or line[0] == '\n': continue
        if line[-1] == '\n': line = line[:-1]
        line = line.strip().split(",")
        features = [float(value) for value in line[:-1]]
        target = line[-1]
        iris_test_data.append((features, target))

# Separate features and target labels
X_iris_test = np.array([data[0] for data in iris_test_data])
Y_iris_test = np.array([data[1] for data in iris_test_data])
Y_iris_test = label_encoder.fit_transform(Y_iris_test) # Transform from strings to ints to avoid errors

# Get the abalone training data
abalone_data = []
with open("abalone.data", "r") as file:
    for line in file:
        if len(line) == 0 or line[0] == '\n': continue
        if line[-1] == '\n': line = line[:-1]
        line = line.strip().split(",")
        features = [float(value) for value in line[:-1]]
        target = line[-1]
        abalone_data.append((features, target))

# Separate features and target labels
X_abalone_train = np.array([data[0] for data in abalone_data])
Y_abalone_train = np.array([data[1] for data in abalone_data])
Y_abalone_train = label_encoder.fit_transform(Y_abalone_train) # Transform from strings to ints to avoid errors

abalone_test_data = []
with open("abalone_test.data", "r") as file:
    for line in file:
        if len(line) == 0 or line[0] == '\n': continue
        if line[-1] == '\n': line = line[:-1]
        line = line.strip().split(",")
        features = [float(value) for value in line[:-1]]
        target = line[-1]
        abalone_test_data.append((features, target))

# Separate features and target labels
X_abalone_test = np.array([data[0] for data in abalone_test_data])
Y_abalone_test = np.array([data[1] for data in abalone_test_data])
Y_abalone_test = label_encoder.fit_transform(Y_abalone_test) # Transform from strings to ints to avoid errors

# Create model for iris, add dense layers one by one specifying activation 
model_iris = Sequential()
model_iris.add(Dense(5, input_dim=4, activation='relu'))
model_iris.add(Dense(5, activation='relu'))
model_iris.add(Dense(4, activation='relu'))
model_iris.add(Dense(1, activation='sigmoid')) # For classification

# Compile the iris model, using adam gradient descent  
model_iris.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

# Train the neural network iris model
model_iris_history = model_iris.fit(X_iris_train,Y_iris_train, epochs = 1000, batch_size=2)

# Evaluate the iris model’s accuracy
score_iris = model_iris.evaluate(X_iris_test, Y_iris_test, verbose=1)
print('Percentage Accuracy on Iris Dataset:', score_iris[1] * 100)

# Use a two dimensional list to enter new data to make a prediction.
X_new_iris = np.array([[4.3,3.2,1.3,0.3]])
iris_prediction = model_iris.predict(X_new_iris)
# print the prediction
print(f"Iris Prediction {iris_prediction}")

# Plot the data score
plt.plot(model_iris_history.history['accuracy'])
plt.plot(model_iris_history.history['loss'])
plt.legend(['train','test'], loc='upper right')
plt.title('Model Error')
plt.ylabel('Iris Avg Absolute Error')
plt.xlabel('Epoch')
plt.show()

dropout_rate = 0.1
# Create model for abalone, add dense layers one by one specifying activation 
model_abalone = Sequential()
model_abalone.add(Dense(8, input_dim=7, activation='relu'))
model_abalone.add(Dropout(dropout_rate))
model_abalone.add(Dense(6, activation='relu'))
model_abalone.add(Dropout(dropout_rate))
model_abalone.add(Dense(1, activation='linear')) # For regression

# Compile the abalone model, using adam gradient descent
model_abalone.compile(loss='mean_squared_error', optimizer="adam", metrics=['mae']) # Mean for regression

# Train the neural network iris model
model_abalone_history = model_abalone.fit(X_abalone_train,Y_abalone_train, epochs = 100, batch_size=2)

# Evaluate the abalone model’s accuracy
score_abalone = model_abalone.evaluate(X_abalone_test, Y_abalone_test, verbose=1)
print('Accuracy on Abalone Dataset:', score_abalone)

# Use a two dimensional list to enter new data to make a prediction.
X_new_abalone = np.array([[0.65,0.53,0.17,0.101,0.037,0.123,0.2]])
abalone_prediction = model_abalone.predict(X_new_abalone)
# print the prediction
print(f"Abalone Prediction {abalone_prediction}")

# Plot the data score
plt.plot(model_abalone_history.history['mae'])
plt.plot(model_abalone_history.history['loss'])
plt.legend(['train','test'], loc='upper right')
plt.title('Model Error')
plt.ylabel('Abalone Mean Absolute Error')
plt.xlabel('Epoch')
plt.show()