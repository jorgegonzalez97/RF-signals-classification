#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Conv1D, Convolution2D, Bidirectional, LSTM, GRU, CuDNNLSTM, MaxPooling1D, Add, AlphaDropout
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, Masking
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from tensorflow.keras.utils import plot_model

import tensorflow as tf
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import json

import scipy.signal as sc
from sklearn.metrics import confusion_matrix
import cmath
import pickle
import scipy.io as sio
import h5py

from tensorflow.keras.callbacks import *
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../../CLR')
from clr_callback import *


# In[2]:
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)

classes = ['LFM','2FSK','4FSK','8FSK', 'Costas','2PSK','4PSK','8PSK','Barker','Huffman','Frank','P1','P2','P3','P4','Px','Zadoff-Chu','T1','T2','T3','T4','NM','ruido']
dt = np.dtype(float)


cw_wv = True

if cw_wv:
    X_train = np.zeros((117300,128,128), dtype=np.float32)
    X_test = np.zeros((78200,128,128), dtype=np.float32)
else:
    X_train = np.zeros((117300,128,128,2), dtype=np.float32)
    X_test = np.zeros((78200,128,128,2), dtype=np.float32)


indx_train = []
indx_test = []

with h5py.File('../../Datasets/radar/tfi_128_s/Xc_train.mat', 'r') as f:
    for i in range(f['Xc_train'].shape[2]):
        #print(i)
        #if i != 1562 and i != 8781:
        try:
            X_train[i,:,:] = f['Xc_train'][:,:,i].T
        except:
            indx_train.append(i)

with h5py.File('../../Datasets/radar/tfi_128_s/Xc_test.mat', 'r') as f:
    for i in range(f['Xc_test'].shape[2]):
        #print(i)
        #if i != 40527 and i != 17204:
        try:
            X_test[i,:,:] = f['Xc_test'][:,:,i].T
        except:
            indx_test.append(i)

indx_train.sort()
indx_test.sort()

print(indx_train)
print(indx_test)

Y_train = sio.loadmat('../../Datasets/radar/tfi_128_s/Y_train.mat')
Y_train = Y_train['Y_train']
Y_test = sio.loadmat('../../Datasets/radar/tfi_128_s/Y_test.mat')
Y_test = Y_test['Y_test']
lbl_train = sio.loadmat('../../Datasets/radar/tfi_128_s/lbl_train.mat')
lbl_train = lbl_train['lbl_train']
lbl_test = sio.loadmat('../../Datasets/radar/tfi_128_s/lbl_test.mat')
lbl_test = lbl_test['lbl_test']


X_train = np.delete(X_train, indx_train, axis = 0)
X_test = np.delete(X_test, indx_test, axis = 0)
Y_train = np.delete(Y_train, indx_train, axis = 0)
Y_test = np.delete(Y_test, indx_test, axis = 0)
lbl_train = np.delete(lbl_train, indx_train, axis = 0)
lbl_test = np.delete(lbl_test, indx_test, axis = 0)


# In[3]:


if cw_wv:
    X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],X_train.shape[2], 1))
    X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],X_test.shape[2], 1))

print("X train shape: ", X_train.shape)
print("X test shape: ", X_test.shape)
print("Y train shape: ", Y_train.shape)
print("Y test shape: ", Y_test.shape)
print("Label train shape: ", lbl_train.shape)
print("Label test shape: ", lbl_test.shape)


# In[6]:


np.random.seed(2020)

X_train, Y_train, lbl_train = shuffle(X_train[:], Y_train[:], lbl_train[:], random_state = 2020)
X_test, Y_test, lbl_test = shuffle(X_test[:], Y_test[:], lbl_test[:], random_state = 2020)


# In[7]:


print(Y_train[:5,:])
print(lbl_train[:5,:])


# In[8]:


print(Y_test[:5,:])
print(lbl_test[:5,:])


# In[9]:
def RecComModel(input_shape):
    """   
    Arguments:
    input_shape -- shape of the inputs of the dataset
        (height, width, channels) as a tuple.  
        Note that this does not include the 'batch' as a dimension.
        If you have a batch like 'X_train', 
        then you can provide the input_shape using
        X_train.shape[1:]
    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input placeholder as a tensor with shape input_shape.
    X_input = Input(input_shape)
        
    X = Conv2D(32, kernel_size=11, padding = 'same',activation='relu')(X_input)
    X = BatchNormalization()(X)
    
    X = MaxPooling2D(2)(X)
    
    X = Conv2D(64, kernel_size=11, padding = 'same',activation='relu')(X)
    X = BatchNormalization()(X)
    
    X = MaxPooling2D(2)(X)
    
    X = Conv2D(128, kernel_size=11, padding = 'same', activation='relu')(X)
    X = BatchNormalization()(X)
    
    X = MaxPooling2D(2)(X)
    
    X = Conv2D(256, kernel_size=11, padding = 'same', activation='relu')(X)
    X = BatchNormalization()(X)
    
    X = MaxPooling2D(2)(X)
    
    X = Conv2D(512, kernel_size=11, padding = 'same', activation='relu')(X)
    X = BatchNormalization()(X)
    
    X = MaxPooling2D(2)(X)
    
    X = Flatten()(X)
    
    X = Dense(512, activation='relu')(X)
    X = Dropout(0.5)(X)

    X = Dense(23, activation='softmax', name='fc0')(X)
    
    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X)
    model.summary()
    
    return model

# In[38]:


model = RecComModel(X_train.shape[1:])


# In[43]:


output_path = '../Results/Radar_RNN/TFI/cw_128_k11/'


clr_triangular = CyclicLR(mode='triangular', base_lr=1e-7, max_lr=1e-4, step_size= 4 * (X_train.shape[0] // 256))

c=[clr_triangular,ModelCheckpoint(filepath= output_path +'best_model.h5', monitor='val_loss', save_best_only=True)]

model.compile(optimizer=optimizers.Adam(1e-7), loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


Train = True

if Train:
    #c = [EarlyStopping(monitor='val_loss', patience=20),
    #            ModelCheckpoint(filepath= output_path +'best_model.h5', monitor='val_loss', save_best_only=True)]

    history = model.fit(X_train, Y_train, epochs = 50, batch_size = 256, callbacks = c, validation_data=(X_test, Y_test))

    with open(output_path +'history_rnn.json', 'w') as f:
        json.dump(history.history, f)
    model_json = model.to_json()
    with open(output_path +'model_rnn.json', "w") as json_file:
        json_file.write(model_json)
else:
    model.load_weights(output_path +'best_model.h5')
    with open(output_path +'history_rnn.json', 'r') as f:
            history = json.load(f)


# In[33]:


model.load_weights(output_path +'best_model.h5')


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['test', 'val'])
plt.show()
plt.savefig(output_path + '\graphs\model_loss.pdf')


# In[ ]:


def getConfusionMatrixPlot(true_labels, predicted_labels,title):
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    cm = np.round(cm_norm,2)
    print(cm)

    # create figure
    width = 18
    height = width / 1.618
    fig = plt.figure(figsize=(width, height))
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    res = ax.imshow(cm, cmap=plt.cm.binary,
                    interpolation='nearest', vmin=0, vmax=1)

    # add color bar
    plt.colorbar(res)

    # annotate confusion entries
    width = len(cm)
    height = len(cm[0])

    for x in range(width):
        for y in range(height):
            ax.annotate(str(cm[x][y]), xy=(y, x), horizontalalignment='center',
                        verticalalignment='center', color=getFontColor(cm[x][y]))

    # add genres as ticks
    alphabet = classes 
    plt.xticks(range(width), alphabet[:width], rotation=30)
    plt.yticks(range(height), alphabet[:height])
    plt.title(title)
    return plt


# In[ ]:


def getFontColor(value):
    if np.isnan(value):
        return "black"
    elif value < 0.2:
        return "black"
    else:
        return "white"



# In[ ]:


acc={}
snrs = [-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,20]
for snr in snrs:
    test_SNRs = list(map(lambda x: lbl_test[x][1], range(0,X_test.shape[0])))
    test_X_i = X_test[[i for i,x in enumerate(test_SNRs) if x==snr]]
    test_Y_i = Y_test[[i for i,x in enumerate(test_SNRs) if x==snr]]       

    # estimate classes
    test_Y_i_hat = np.array(model.predict(test_X_i))
    width = 18
    height = width / 1.618
    plt.figure(figsize=(width, height))
    plt = getConfusionMatrixPlot(np.argmax(test_Y_i, 1), np.argmax(test_Y_i_hat, 1),title="Confusion Matrix (SNR=%d)"%(snr))
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig(output_path + '\graphs\confmat_'+str(snr)+'.pdf')
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])
    for i in range(0,test_X_i.shape[0]):
        j = list(test_Y_i[i,:]).index(1)
        k = int(np.argmax(test_Y_i_hat[i,:]))
        conf[j,k] = conf[j,k] + 1 
    for i in range(0,len(classes)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    plt.figure()
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor 
    print("Overall Accuracy: ", cor / (cor+ncor))
    acc[snr] = 1.0*cor/(cor+ncor)
#print(acc)


# In[ ]:


with open(output_path +'acc.json', 'w') as f:
        json.dump(acc, f)
        
plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("Classification Accuracy on Radar Dataset")
plt.savefig(output_path + '\graphs\clas_acc.pdf')


# In[ ]:





# In[ ]:





# In[ ]:




