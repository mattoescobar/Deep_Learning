# TEP Fault Detection 
ANN and RNN applied to the Tennessee Eastman Process (TEP) dataset. More details on TEP can be found on this repository's wiki. 

## Prerequisites
* TensorFlow Framework
* Keras API

## Usage
### Comparison between ANN and RNN's fault detection performance - roc_ann_rnn.py
```python
# Comparison of ANN and RNN performances using a ROC curve
import numpy as np
import matplotlib.pyplot as plt

# Load ann and rnn roc related data
ann_data = np.load('ann_roc.npz')
ann_fpr = ann_data['arr_0']
ann_tpr = ann_data['arr_1']
ann_roc = ann_data['arr_2']
rnn_data = np.load('rnn_roc.npz')
rnn_fpr = rnn_data['arr_0']
rnn_tpr = rnn_data['arr_1']
rnn_roc = rnn_data['arr_2']

# Plot ROC curves for both ann and rnn fault detection tasks
plt.title('Receiver Operating Characteristic')
plt.plot(ann_fpr, ann_tpr, 'b', label = 'AUC_ANN = %0.2f' % ann_roc)
plt.plot(rnn_fpr, rnn_tpr, 'r', label='AUC_RNN = %0.2f' % rnn_roc)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
```

