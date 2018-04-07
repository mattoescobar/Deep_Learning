# Artificial Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio

# Importing the training set
dataset = sio.loadmat('TEP_9.mat')
dataset_train = np.concatenate([dataset['X0'], dataset['X']], axis=0)
y_target_train = np.zeros(dataset_train.shape[0])
y_target_train[480:] = 1

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(dataset_train)

X_train, y_train = np.array(training_set_scaled), np.array(y_target_train)

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = X_train.shape[1]))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)


# Part 3 - Making the predictions and visualising the results

# Getting the test target values
dataset_test = dataset['Xt']
y_target_test = np.zeros(dataset_test.shape[0])
y_target_test[160:] = 1

# Visualising test data
# plt.figure()
# plt.plot(time_test, dataset_test[:,0])
# plt.show()

# Feature Scaling and data structure
test_set_scaled = sc.transform(dataset_test)
X_test = np.array(test_set_scaled)
y_test = np.array(y_target_test)
predicted_state = classifier.predict(X_test)

# Visualising the results
plt.plot(y_test, color = 'red', label = 'TEP anomaly label')
plt.plot(predicted_state, color = 'blue', label = 'TEP fault detection')
plt.title('TEP fault detection ')
plt.xlabel('Time')
plt.ylabel('Anomalous behaviour likelihood')
plt.legend()
plt.show()

# Visualising performance through ROC curve
import sklearn.metrics as skm
fpr, tpr, threshold = skm.roc_curve(y_test, predicted_state)
roc_auc = skm.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
