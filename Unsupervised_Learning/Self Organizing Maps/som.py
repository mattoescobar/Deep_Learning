# Self Organizing Map

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM 100 times to spot frauds more consistently
from minisom import MiniSom
frauds = np.array(None)
for i in range(100):
    som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
    som.random_weights_init(X)
    som.train_random(data = X, num_iteration = 100)

    # # Visualizing the results
    # from pylab import bone, pcolor, colorbar, plot, show
    #
    # bone()
    # pcolor(som.distance_map().T)
    # colorbar()
    # markers = ['o', 's']
    # colors = ['r', 'g']
    # for i, x in enumerate(X):
    #     w = som.winner(x)
    #     plot(w[0] + 0.5,
    #          w[1] + 0.5,
    #          markers[y[i]],
    #          markeredgecolor=colors[y[i]],
    #          markerfacecolor='None',
    #          markersize=10,
    #          markeredgewidth=2)
    # show()

    # Finding the frauds
    mappings = som.win_map(X)

    # Finding the coordinates with maximum normalised distance -> Outliers
    outlier_mappings = np.unravel_index(som.distance_map().T.argmax(), som.distance_map().T.shape)
    outlier_mappings = outlier_mappings[::-1]

    frauds_now = mappings[(outlier_mappings[0], outlier_mappings[1])]

    if not frauds:
        frauds = frauds_now
    else:
        if frauds_now:
            frauds = np.concatenate((frauds, frauds_now), axis=0).tolist()
    print(i)

frauds = np.unique(frauds, axis=0)
frauds = sc.inverse_transform(frauds)


