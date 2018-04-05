# Dimensionality reduction comparison between GTM, SOM, and AE

# Importing the libraries
from GTM_Indexes import GTMIndexes
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing a spherical data set
spherical_data = np.load('Spherical_data.npz')
spherical_input_data = spherical_data['arr_0']
spherical_input_test_data = spherical_data['arr_1']
t = np.zeros(2000)
t[1000:] = 1

# -------------- GTM Training ---------------
test = GTMIndexes(spherical_input_data, latent_space_size=3600, rbf_number=64, regularization=0.001, rbf_width=2,
                  iterations=50)
[w_optimal, beta_optimal, log_likelihood_evolution] = test.gtm_training()

# Visualising GTM
fig1 = plt.figure()
means2 = test.gtm_mean(w_optimal, beta_optimal)
plt.scatter(means2[:, 0], means2[:, 1], c=t)
plt.show()

# -------------- SOM Training ---------------
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(spherical_input_data)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x=60, y=60, input_len=3, sigma=10.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)

# Finding the winner neurons
w = []
for i, x in enumerate(X):
    w.append(som.winner(x))
w = np.array(w)

# Visualising SOM
plt.scatter(w[:, 0] + 0.5, w[:, 1] + 0.5, c=t)
plt.show()

# -------------- AE Training ---------------
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.preprocessing import StandardScaler

# Preparing the datasets
sc = StandardScaler()
X_train = sc.fit_transform(spherical_input_data)
# Encoding to a 2D map representation
encoding_dim = 2
# Input placeholder
input_data = Input(shape=(3,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_data)
# "decoded" is the loss reconstruction of the input
decoded = Dense(3, activation='sigmoid')(encoded)
# Mapping inputs to their reconstruction
autoencoder = Model(input_data, decoded)
# Mapping inputs to their encoded representation
encoder = Model(input_data, encoded)

# Training AE
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.fit(X_train, X_train, epochs=100, batch_size=50, shuffle=True, validation_data=(X_train, X_train))

# Visualising AE encoding map
encoded_imgs = encoder.predict(X_train)
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=t)
plt.show()


