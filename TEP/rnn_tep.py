# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio

# Importing the training set
dataset = sio.loadmat('TEP_21.mat')
dataset_train = np.concatenate([dataset['X0'], dataset['X']], axis=0)
time_train = np.array(range(0, dataset_train.shape[0]))
y_target_train = np.zeros(dataset_train.shape[0])
y_target_train[480:] = 1

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(dataset_train)

# # Visualising TEP faults
# plt.figure()
# plt.plot(time_train, training_set_scaled)
# plt.show()

# Creating a data structure with X timesteps and 1 output
X_train = []
y_train = []
time_step = 15
for i in range(time_step, training_set_scaled.shape[0]):
    X_train.append(training_set_scaled[i-time_step:i, :])
    y_train.append(y_target_train[i])
X_train, y_train = np.array(X_train), np.array(y_train)

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 10, batch_size = 32)



# Part 3 - Making the predictions and visualising the results

# Getting the test target values
dataset_test = np.concatenate([dataset_train[0:time_step, :], dataset['Xt']], axis=0)
time_test = np.array(range(0, dataset_test.shape[0]))
y_target_test = np.zeros(dataset_test.shape[0])
y_target_test[160+time_step:] = 1

# Visualising test data
# plt.figure()
# plt.plot(time_test, dataset_test[:,0])
# plt.show()

# Feature Scaling and data structure
test_set_scaled = sc.transform(dataset_test)
X_test = []
y_test = []
for i in range(time_step, dataset_test.shape[0]):
    X_test.append(test_set_scaled[i-time_step:i, :])
    y_test.append(y_target_test[i])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

predicted_state = regressor.predict(X_test)


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
