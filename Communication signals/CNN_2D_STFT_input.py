#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from keras import layers, Sequential
from keras import optimizers, Input
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Conv1D, Convolution2D, Bidirectional, LSTM, GRU
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D,Add, MaxPooling1D, AlphaDropout
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.applications import Xception

import tensorflow as tf
import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping, ModelCheckpoint
import json

import scipy.signal as sc
from sklearn.metrics import confusion_matrix
import cmath
import pickle
import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# In[2]:


import h5py

filename = "../../Datasets/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5"

f = h5py.File(filename, 'r')

X = f['X']
Y = f['Y']
SNR = f['Z']

classes = ['OOK', '4ASK', '8ASK','BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK', '64APSK', '128APSK','16QAM', '32QAM', '64QAM', '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']
snrs = [-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]

with open("../../Datasets/2018.01/lbl.txt", "rb") as fp:
    lbl = pickle.load(fp)


# In[3]:

np.random.seed(2018)

X, Y, lbl = shuffle(X[:], Y[:], lbl[:], random_state = 2018)

n_examples = X.shape[0]
n_train = int(n_examples * 0.5)

X_train = X[:n_train]
Y_train = Y[:n_train]
lbl_train = lbl[:n_train]

X_test =  X[n_train:]
Y_test =  Y[n_train:]
lbl_test = lbl[n_train:]


# In[4]:

print('1')

X_tr = np.zeros((n_train, 256, 9, 2), dtype=np.float16)
X_te = np.zeros((n_train, 256, 9, 2), dtype=np.float16)

print('2')
for i in range(X_train.shape[0]):
    X_train_i = X_train[i,:,0] + 1j * X_train[i,:,1] 
    X_test_i = X_test[i,:,0] + 1j * X_test[i,:,1]
    
    f, t, z = sc.stft(X_train_i, nperseg = 256, noverlap = 128)
    
    X_tr[i,:,:, 0] = np.abs(z)
    X_tr[i,:,:, 1] = np.arctan2(np.imag(z),np.real(z))/np.pi
    
    f, t, z = sc.stft(X_test_i, nperseg = 256, noverlap = 128)
    
    X_te[i,:,:, 0] = np.abs(z)
    X_te[i,:,:, 1] = np.arctan2(np.imag(z),np.real(z))/np.pi

print('3')
X_train = X_tr
X_test = X_te

del X_tr
del X_te

print('4')
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

    X = Conv2D(32, 2)(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = Conv2D(32, 3, padding = 'same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = Conv2D(32, 3, padding = 'same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = MaxPooling2D((2,1))(X)
    
    X = Conv2D(64, 3,  padding = 'same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = Conv2D(64, 3,  padding = 'same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = Conv2D(64, 3,  padding = 'same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = MaxPooling2D((2,1))(X)
    
    X = Conv2D(128, 3,  padding = 'same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = Conv2D(128, 3,  padding = 'same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = Conv2D(128, 3,  padding = 'same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = MaxPooling2D((2,1))(X)
    
    X = Conv2D(256, 3,  padding = 'same')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(256, 3,  padding = 'same')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(512, 3,  padding = 'same')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(512, 3,  padding = 'same')(X)
    X = Activation('relu')(X)

    X = Flatten()(X)
    
    X = Dense(256, activation='relu')(X)
    X = Dropout(0.5)(X)
    
    X = Dense(256, activation='relu')(X)
    X = Dropout(0.5)(X)

    X = Dense(24, activation='softmax')(X)
    
    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X)
    model.summary()
    
    return model


# In[8]:

model = RecComModel(X_train.shape[1:])
print('5')
opt = optimizers.Adam(0.0001)

model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ["accuracy"])
print('6')

# In[9]:


output_path = '../Results/CONV_stft_256/'


# In[10]:


Train = True

if Train:
    c = [EarlyStopping(monitor='val_loss', patience=10),
                ModelCheckpoint(filepath= output_path +'best_model.h5', monitor='val_loss', save_best_only=True)]

    history = model.fit(X_train, Y_train, shuffle = True, batch_size = 16, callbacks = c, validation_data=(X_test, Y_test), epochs = 100)

    with open(output_path +'history_rnn.json', 'w') as f:
        json.dump(history.history, f)
    model_json = model.to_json()
    with open(output_path +'model_rnn.json', "w") as json_file:
        json_file.write(model_json)
else:
    model.load_weights(output_path +'best_model.h5')
 #   with open(output_path +'history.pickle', 'rb') as f:
 #       history = pickle.load(f)


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
    width = 16.4 
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
for snr in snrs:
    test_SNRs = list(map(lambda x: lbl_test[x][1], range(0,n_train)))
    test_X_i = X_test[[i for i,x in enumerate(test_SNRs) if x==snr]]
    test_Y_i = Y_test[[i for i,x in enumerate(test_SNRs) if x==snr]]       

    # estimate classes
    test_Y_i_hat = np.array(model.predict(test_X_i))
    width = 16.4 
    height = width / 1.618
    plt.figure(figsize=(width, height))
    plt = getConfusionMatrixPlot(np.argmax(test_Y_i, 1), np.argmax(test_Y_i_hat, 1),title="RecNet Confusion Matrix (SNR=%d)"%(snr))
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig(output_path + '\graphs\confmat_'+str(snr)+'.pdf')
    conf = np.zeros([len(snrs),len(snrs)])
    confnorm = np.zeros([len(snrs),len(snrs)])
    for i in range(0,test_X_i.shape[0]):
        j = list(test_Y_i[i,:]).index(1)
        k = int(np.argmax(test_Y_i_hat[i,:]))
        conf[j,k] = conf[j,k] + 1 
    for i in range(0,len(snrs)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    plt.figure()
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor 
    print("Overall Accuracy: ", cor / (cor+ncor))
    acc[snr] = 1.0*cor/(cor+ncor)
#print(acc)


# In[ ]:


plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("Classification Accuracy on RadioML 2018.10 Alpha")
plt.savefig(output_path + '\graphs\clas_acc.pdf')


with open(output_path +'acc.json', 'w') as f:
        json.dump(acc, f)

# In[ ]:

# Plot confusion matrix
for mod in classes:
    acc = {}
    for snr in snrs:
      # extract classes @ SNR
        dev_MS = list(map(lambda x: [lbl_test[x][0],lbl_test[x][1]], range(0,n_train)))
        dev_X_i = X_test[[i for i,x in enumerate(dev_MS) if (x[0]==mod and x[1]==snr)]]
        dev_Y_i = Y_test[[i for i,x in enumerate(dev_MS) if (x[0]==mod and x[1]==snr)]]     
      # estimate classes
        pred = model.evaluate(dev_X_i, dev_Y_i, batch_size=1024, verbose=0)
        acc[snr] = pred[1]
    plt.figure()
    plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
    #plt.plot(snrs, 0.9*np.ones((len(snrs),)), '--r')
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Accuracy")
    plt.ylim(-0.05, 1.05)
    plt.title("Classification Accuracy (MOD=" + mod + ")")
    plt.show()
    plt.savefig(output_path + '\graphs\confmat_'+mod+'.pdf')




# In[ ]:





# In[ ]:




