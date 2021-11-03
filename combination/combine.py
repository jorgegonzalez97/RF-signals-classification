#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Conv1D, Convolution2D, Bidirectional, LSTM, GRU, CuDNNLSTM
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


# In[3]:


classes2 = ['LFM','2FSK','4FSK','8FSK', 'Costas','2PSK','4PSK','8PSK','Barker','Huffman','Frank','P1','P2','P3','P4','Px','Zadoff-Chu','T1','T2','T3','T4','NM','noise']
classes1 = ['OOK', '4ASK', '8ASK','BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK', '64APSK', '128APSK','16QAM', '32QAM', '64QAM', '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']

snrs = [-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,20]

classes = classes2+classes1
print(classes)
print(len(classes))


# In[4]:


dt = np.dtype(float)
indx_test = []
indx_train = []

with h5py.File('../../Datasets/radar/Interpolation_orthogonal/X_train.mat', 'r') as f:
    X_train = np.zeros((f['X_train'].shape[2],1024,2))
    for i in range(f['X_train'].shape[2]):
        try:
            X_train[i,:,:] = f['X_train'][:,:,i].T
        except:
            indx_train.append(i)
    
with h5py.File('../../Datasets/radar/Interpolation_orthogonal_px_odd/X_test.mat', 'r') as f:
    X_test = np.zeros((f['X_test'].shape[2],1024,2))
    for i in range(f['X_test'].shape[2]):
        try:
            X_test[i,:,:] = f['X_test'][:,:,i].T
        except:
            indx_test.append(i)
    
indx_test.sort()
indx_train.sort()
print(indx_train)
print(indx_test)


Y_train = sio.loadmat('../../Datasets/radar/Interpolation_orthogonal/Y_train.mat')
Y_train = Y_train['Y_train']
Y_test = sio.loadmat('../../Datasets/radar/Interpolation_orthogonal_px_odd/Y_test.mat')
Y_test = Y_test['Y_test']
lbl_train = sio.loadmat('../../Datasets/radar/Interpolation_orthogonal/lbl_train.mat')
lbl_train = lbl_train['lbl_train']
lbl_test = sio.loadmat('../../Datasets/radar/Interpolation_orthogonal_px_odd/lbl_test.mat')
lbl_test = lbl_test['lbl_test']

X_train = np.delete(X_train, indx_train, axis = 0)
Y_train = np.delete(Y_train, indx_train, axis = 0)
lbl_train = np.delete(lbl_train, indx_train, axis = 0)

X_test = np.delete(X_test, indx_test, axis = 0)
Y_test = np.delete(Y_test, indx_test, axis = 0)
lbl_test = np.delete(lbl_test, indx_test, axis = 0)


# In[5]:


lbl_train = lbl_train[:,:2]
lbl_test = lbl_test[:,:2]


# In[6]:


print(lbl_test)


# In[7]:


print("X train shape: ", X_train.shape)
print("X test shape: ", X_test.shape)
print("Y train shape: ", Y_train.shape)
print("Y test shape: ", Y_test.shape)
print("Label train shape: ", lbl_train.shape)
print("Label test shape: ", lbl_test.shape)


# In[8]:


import h5py

filename = "../../Datasets/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5"

f = h5py.File(filename, 'r')

X = f['X']
Y = f['Y']
SNR = f['Z']



with open("../../Datasets/2018.01/lbl.txt", "rb") as fp:
    lbl2 = pickle.load(fp)

idx_snrs = []
lbl = []
for i in range(len(SNR)):
    if SNR[i][0] in snrs:
        idx_snrs.append(i)
        lbl.append([classes.index(lbl2[i][0])+1,lbl2[i][1]])


# In[9]:


len(idx_snrs)


# In[10]:


print("X shape: ", X.shape)
print("Y shape: ", Y.shape)


# In[11]:


X = X[idx_snrs]
Y = Y[idx_snrs]
#SNR = SNR[idx_snrs]


# In[12]:


print("X shape: ", X.shape)
print("Y shape: ", Y.shape)


# In[13]:


X, Y, lbl = shuffle(X[:], Y[:], lbl[:], random_state = 2018)

n_examples = X.shape[0]
n_train = int(n_examples * 0.4)

X_train2 = X[:n_train]
Y_train2 = Y[:n_train]
lbl_train2 = lbl[:n_train]

X_test2 =  X[n_train:]
Y_test2 =  Y[n_train:]
lbl_test2 = lbl[n_train:]


# In[17]:


X_train = np.concatenate((X_train,X_train2), axis=0).astype('float32')
X_test = np.concatenate((X_test,X_test2), axis=0).astype('float32')


# In[18]:


lbl_train = np.concatenate((lbl_train,lbl_train2), axis=0).astype('float16')
lbl_test = np.concatenate((lbl_test,lbl_test2), axis=0).astype('float16')


# In[19]:


Y_tr = np.zeros((X_train.shape[0], len(classes)),dtype='float16')
Y_te = np.zeros((X_test.shape[0], len(classes)),dtype='float16')

Y_tr[:Y_train.shape[0], :len(classes2)] = Y_train[:,:]
Y_tr[Y_train.shape[0]:, len(classes2):] = Y_train2[:,:]

Y_te[:Y_test.shape[0], :len(classes2)] = Y_test[:,:]
Y_te[Y_test.shape[0]:, len(classes2):] = Y_test2[:,:]


# In[20]:


Y_train = Y_tr
Y_test = Y_te

del Y_tr, Y_train2, Y_te, Y_test2, X, Y, SNR, lbl2, X_train2, X_test2, lbl_train2, lbl_test2


# In[21]:


print("X train shape: ", X_train.shape)
print("X test shape: ", X_test.shape)
print("Y train shape: ", Y_train.shape)
print("Y test shape: ", Y_test.shape)
print("Label train shape: ", lbl_train.shape)
print("Label test shape: ", lbl_test.shape)


# In[22]:


np.random.seed(2020)

X_train, Y_train, lbl_train = shuffle(X_train[:], Y_train[:], lbl_train[:], random_state = 2020)
X_test, Y_test, lbl_test = shuffle(X_test[:], Y_test[:], lbl_test[:], random_state = 2020)


# In[23]:


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
    
    #X = Masking(mask_value=0.)(X_input)

    X = CuDNNLSTM(128, return_sequences= True, name = 'lstm0')(X_input)

    X = CuDNNLSTM(128, return_sequences= True, name = 'lstm1')(X)
    
    X = CuDNNLSTM(128, return_sequences= False, name = 'lstm2')(X)

   # X = Flatten()(X)

    X = Dense(47, activation='softmax', name='fc0')(X)
    
    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X)
    model.summary()
    
    return model


# In[24]:


model = RecComModel(X_train.shape[1:])


# In[25]:


output_path = '../Results/COMBINE2/'

clr_triangular = CyclicLR(mode='triangular', base_lr=0.0000001, max_lr=0.0001, step_size= 4 * (X_train.shape[0] // 1200))

c=[clr_triangular,ModelCheckpoint(filepath= output_path +'best_model.h5', monitor='val_loss', save_best_only=True)]

model.compile(optimizer=optimizers.Adam(1e-7), loss='categorical_crossentropy', metrics=['accuracy'])



# In[27]:


input_path = '../Results/COMBINE/'
model.load_weights(input_path +'best_model.h5')

# In[28]:


Train = True

if Train:
    #c = [EarlyStopping(monitor='val_loss', patience=20),
    #            ModelCheckpoint(filepath= output_path +'best_model.h5', monitor='val_loss', save_best_only=True)]

    history = model.fit(X_train, Y_train, epochs = 200, batch_size = 512, callbacks = c, validation_data=(X_test, Y_test))

    with open(output_path +'history_rnn.json', 'w') as f:
        json.dump(history.history, f)
    model_json = model.to_json()
    with open(output_path +'model_rnn.json', "w") as json_file:
        json_file.write(model_json)
else:
    model.load_weights(output_path +'best_model.h5')
    with open(output_path +'history_rnn.json', 'r') as f:
            history = json.load(f)


# In[ ]:

model.load_weights(output_path +'best_model.h5')
output_path = '../Results/COMBINE2/'


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'])
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
    plt = getConfusionMatrixPlot(np.argmax(test_Y_i, 1), np.argmax(test_Y_i_hat, 1),title="RecNet Confusion Matrix (SNR=%d)"%(snr))
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


# Plot confusion matrix
for mod in classes:
    mod_n = classes.index(mod) + 1
    acc = {}
    for snr in snrs:
      # extract classes @ SNR
        dev_MS = list(map(lambda x: [lbl_test[x][0],lbl_test[x][1]], range(0,X_test.shape[0])))
        dev_X_i = X_test[[i for i,x in enumerate(dev_MS) if (x[0]==mod_n and x[1]==snr)]]
        dev_Y_i = Y_test[[i for i,x in enumerate(dev_MS) if (x[0]==mod_n and x[1]==snr)]]     
      # estimate classes
        pred = model.evaluate(dev_X_i, dev_Y_i, verbose=0)
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


# Plot confusion matrix
acc_mods = {}
for mod in classes:
    mod_n = classes.index(mod) + 1
    acc = {}
    for snr in snrs:
      # extract classes @ SNR
        dev_MS = list(map(lambda x: [lbl_test[x][0],lbl_test[x][1]], range(0,X_test.shape[0])))
        dev_X_i = X_test[[i for i,x in enumerate(dev_MS) if (x[0]==mod_n and x[1]==snr)]]
        dev_Y_i = Y_test[[i for i,x in enumerate(dev_MS) if (x[0]==mod_n and x[1]==snr)]]     
      # estimate classes
        pred = model.evaluate(dev_X_i, dev_Y_i, batch_size=1024, verbose=0)
        acc[snr] = pred[1]
    acc_mods[mod] = acc
    #plt.figure()
    #plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
    #plt.plot(snrs, 0.9*np.ones((len(snrs),)), '--r')
    #plt.xlabel("Signal to Noise Ratio")
    #plt.ylabel("Accuracy")
    #plt.ylim(-0.05, 1.05)
    #plt.title("Classification Accuracy (MOD=" + mod + ")")
    #plt.show()
    #plt.savefig(output_path + '\graphs\confmat_'+mod+'.pdf')

with open(output_path +'acc_mods.json', 'w') as f:
        json.dump(acc_mods, f)


# In[ ]:


mods_cont = ['LFM','2FSK','4FSK','8FSK','2PSK','4PSK','8PSK','NM','noise']
mods_puls = ['Costas','Barker','Huffman','Frank','P1','P2','P3','P4','Px','Zadoff-Chu','T1','T2','T3','T4']

mods_psk = ['2PSK','4PSK','8PSK']
mods_fsk = ['2FSK','4FSK','8FSK']
mods_p = ['P1','P2','P3','P4','Px']
mods_t = ['T1','T2','T3','T4']


for mod in mods_cont:
    acc = acc_mods[mod]
    plt.plot(snrs, list(map(lambda x: acc[x], snrs)))

plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Accuracy")
plt.ylim(-0.05, 1.05)
plt.title("Classification Accuracy on continuous signals")
plt.legend(mods_cont)
plt.show()

for mod in mods_puls:
    acc = acc_mods[mod]
    plt.plot(snrs, list(map(lambda x: acc[x], snrs)))

plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Accuracy")
plt.ylim(-0.05, 1.05)
plt.title("Classification Accuracy on pulsed signals")
plt.legend(mods_puls)
plt.show()

for mod in mods_psk:
    acc = acc_mods[mod]
    plt.plot(snrs, list(map(lambda x: acc[x], snrs)))

plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Accuracy")
plt.ylim(-0.05, 1.05)
plt.title("Classification Accuracy on PSK signals")
plt.legend(mods_psk)
plt.show()

for mod in mods_fsk:
    acc = acc_mods[mod]
    plt.plot(snrs, list(map(lambda x: acc[x], snrs)))

plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Accuracy")
plt.ylim(-0.05, 1.05)
plt.title("Classification Accuracy on FSK signals")
plt.legend(mods_fsk)
plt.show()

for mod in mods_p:
    acc = acc_mods[mod]
    plt.plot(snrs, list(map(lambda x: acc[x], snrs)))

plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Accuracy")
plt.ylim(-0.05, 1.05)
plt.title("Classification Accuracy on Radar dataset")
plt.legend(mods_p)
plt.show()

for mod in mods_t:
    acc = acc_mods[mod]
    plt.plot(snrs, list(map(lambda x: acc[x], snrs)))

plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Accuracy")
plt.ylim(-0.05, 1.05)
plt.title("Classification Accuracy on Radar dataset")
plt.legend(mods_t)
plt.show()


# In[ ]:


plt.figure(figsize=(10,10))
for mod in mods_psk:
    acc = acc_mods[mod]
    plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
for mod in mods_apsk:
    acc = acc_mods[mod]
    plt.plot(snrs, list(map(lambda x: acc[x], snrs)),'--')
for mod in mods_qam:
    acc = acc_mods[mod]
    plt.plot(snrs, list(map(lambda x: acc[x], snrs)),':')
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Accuracy")
plt.ylim(-0.05, 1.05)
plt.title("Classification Accuracy on Radar dataset")
plt.legend(mods_psk+mods_apsk+mods_qam)
plt.show()


# In[ ]:





# In[ ]:




