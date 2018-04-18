# Deep_Learning
Getting acquainted to deep learning methodologies and frameworks. 

Most of what is presented here derives from the data repository provided by the SuperDataScience Deep Learning course. More details can be found in the URL: [http://www.superdatascience.com/deep-learning]

This repository includes codes and implementations on the following topics: <br/>
- Supervised Learning: Artificial (ANN), Convolutional (CNN), and Recurrent Neural Networks (RNN) <br/>
- Unsupervised Learning: Self-Organising Maps (SOM), Boltzmann Machines, and AutoEncoders (AE)

The SOM sub-directory also includes a hybrid implementation (hybrid_fault_detection.py), where SOM is initially used to discriminate samples between two classes and then ANN is trained to predict the probability of new samples belonging to each class. 

TEP subdirectory shows ANN and RNN applied to the Tennessee Eastman Process dataset. This industrial dataset is used for the development of monitoring methodologies used for the detection of online anonalous behaviour. More details on TEP an be found on this repository's wiki.   

## Prerequisites
* TensorFlow and PyTorch frameworks
* Keras high level API
* Scikit-learn

## License
Please refer to LICENSE in the root of this directory 
