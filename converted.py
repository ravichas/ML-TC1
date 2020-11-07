from __future__ import print_function
import os, sys, gzip, glob, json, time, argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from pandas.io.json import json_normalize
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from keras.utils import to_categorical
from keras import backend as K
from keras.layers import Input, Dense, Dropout, Activation, Conv1D, MaxPooling1D, Flatten
from keras import optimizers
from keras.optimizers import SGD, Adam, RMSprop
from keras.models import Sequential, Model, model_from_json, model_from_yaml
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from keras.callbacks import EarlyStopping

TC1data3 = pd.read_csv("Data/TC1-data3stypes.tsv", sep="\t", low_memory = False)
outcome = pd.read_csv("Data/TC1-outcome-data3stypes.tsv", sep="\t", low_memory=False, header=None)

TC1data3.iloc[[0,1,2,3,4],[0,1,2,3,4,5,6,7,8,9,60400,60401,60482]]

outcome = outcome[0].values

def encode(data):
    print('Shape of data (BEFORE encode): %s' % str(data.shape))
    encoded = to_categorical(data)
    print('Shape of data (AFTER  encode): %s\n' % str(encoded.shape))
    return encoded

outcome = encode(outcome)

# from IPython.core.display import Image
# Image(filename='Img/Train-Test.png',width = 600, height = 800 )

X_train, X_test, Y_train, Y_test = train_test_split(TC1data3, outcome,
                                                    train_size=0.75,
                                                    test_size=0.25,
                                                    random_state=123,
                                                    stratify = outcome)

activation='relu'
batch_size=20
classes=3

drop = 0.1
feature_subsample = 0
loss='categorical_crossentropy'

out_act='softmax'

shuffle = False

epochs=5
optimizer = optimizers.SGD(lr=0.1)
metrics = ['acc']


x_train_len = X_train.shape[1]

X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

filters = 128
filter_len = 20
stride = 1

# K.clear_session()

# from IPython.core.display import Image
# Image(filename='Img/TC1-arch.png',width = 300, height = 400 )


model = Sequential()

model.add(Conv1D(filters = filters,
                 kernel_size = filter_len,
                 strides = stride,
                 padding='valid',
                 input_shape=(x_train_len, 1)))

model.add(Activation('relu'))

model.add(MaxPooling1D(pool_size = 1))

model.add(Conv1D(filters=filters,
                 kernel_size=filter_len,
                 strides=stride,
                 padding='valid'))

model.add(Activation('relu'))

model.add(MaxPooling1D(pool_size = 10))

model.add(Flatten())

model.add(Dense(200))

model.add(Activation('relu'))

model.add(Dropout(0.1))

model.add(Dense(20))

model.add(Activation('relu'))

model.add(Dropout(0.1))

model.add(Dense(3))

model.add(Activation(out_act))

model.compile( loss= loss,
              optimizer = optimizer,
              metrics = metrics )

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
model.summary()


save = '.'
output_dir = "Model"

if not os.path.exists(output_dir):
        os.makedirs(output_dir)

model_name = 'tc1'
path = '{}/{}.autosave.model.h5'.format(output_dir, model_name)
checkpointer = ModelCheckpoint(filepath=path,
                               verbose=1,
                               save_weights_only=True,
                               save_best_only=True)

csv_logger = CSVLogger('{}/training.log'.format(output_dir))


reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.1,
                              patience=10,
                              verbose=1, mode='auto',
                              min_delta=0.0001,
                              cooldown=0,
                              min_lr=0)

history = model.fit(X_train, Y_train, batch_size=batch_size,
                    epochs=epochs, verbose=1, validation_data=(X_test, Y_test),
                    callbacks = [checkpointer, csv_logger, reduce_lr])


# # %%
# score = model.evaluate(X_test, Y_test, verbose=0)

# print('Test score:', score[0])
# print('Test accuracy:', score[1])

# # %% [markdown]
# # ## Word of caution about the accuracy
# # 
# # The output loss and accuracy from smalller sample sizes (for example, n = 50) will not reflect the real learning. For good accuracy, we need to use the whole dataset. Here are few epochs from the original dataset modeling (Train: 3375; Validate: 1125).

# # %%
# from IPython.core.display import Image
# Image(filename='Img/TC1-Acc.PNG',width = 1000, height = 1000 )


# # %%
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# tc1results = pd.read_csv("Output/tc1results.txt", index_col='epoch')


# # %%
# tc1results.plot()

# # %% [markdown]
# # ## How to save the model/weights?

# # %%
# # JSON JSON
# # serialize model to json
# json_model = model.to_json()

# # save the model architecture to JSON file
# with open('Model/tc1.model.json', 'w') as json_file:
#     json_file.write(json_model)


# # YAML YAML
# # serialize model to YAML
# model_yaml = model.to_yaml()

# # save the model architecture to YAML file
# with open("{}/{}.model.yaml".format(output_dir, model_name), "w") as yaml_file:
#     yaml_file.write(model_yaml)


# # WEIGHTS HDF5
# # serialize weights to HDF5
# model.save_weights("{}/{}.model.h5".format(output_dir, model_name))
# print("Saved model to disk")

# # %% [markdown]
# # ## Inference
# # 
# # The calculation was carried out on a NIH Biowulf GPU node. Model weights were saved in Python HDF5 grid format. HDF5 is ideal for storing multi-dimensional arrays of numbers. You can read about HDF5 here.
# # http://www.h5py.org/

# # %%
# from keras.models import model_from_json

# # Open the handle
# json_file = open('Model/tc1.model.json', 'r')

# # load json and create model
# loaded_model_json = json_file.read()
# json_file.close()

# loaded_model = model_from_json(loaded_model_json)

# # load weights into new model
# loaded_model.load_weights('Model/tc1.model.h5')
# print("Loaded model from disk")
# # loaded_model_json

# # %% [markdown]
# # ## Mimicking the process of external set
# # 
# # Note this is a demonstration of how to use external data for inference.  
# # 
# # When you bring in an external dataset. Make sure you follow the following steps:
# # 
# # a) Make sure you do the same operations that you had done to the data set 
# # b) scale the inference dataset in the same way as the training data 
# # 

# # %%
# import numpy as np
# chosen_idx = np.random.choice(38, replace=False, size=5)
# # X_test[chosen_idx].shape
# # Y_test[chosen_idx].shape
# # Y_test.shape


# # %%
# X_mini = X_test[chosen_idx]
# y_mini = Y_test[chosen_idx]
# # df_trimmed = X_mini.drop(X_mini.columns[[0]], axis=1, inplace=False)
# # X_mini = df_trimmed
# print('X_mini.shape', X_mini.shape)
# print('len(y_minip)', len(y_mini))


# # %%
# print('X_mini.shape', X_mini.shape)
# print('y_mini.shape', y_mini.shape)


# # %%
# # evaluate loaded model on test data
# loaded_model.compile(loss='categorical_crossentropy', optimizer='sgd', 
#                      metrics=['accuracy'])
# score = loaded_model.evaluate(X_mini, y_mini, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

# # %% [markdown]
# # ## Unsupervised learning plots (PCA and tSNE)
# # 
# # ### *This section was based on Dr. Andrew Weissman's code template. Check out Andrew's Gihub here, https://github.com/andrew-weisman*
# # %% [markdown]
# # ### Load the custom `tc1_library.py` file, which contains the unsupervised learning plotting function

# # %%
# import tc1_library
# import importlib

# importlib.reload(tc1_library);

# # %% [markdown]
# # ### Decode the outcome matrix back into a single vector (remember earlier that we one-hot-encoded it)

# # %%
# outcome[0:5]

# # %% [markdown]
# # ##### Let us explore some outcome values

# # %%
# outcome.shape # 150,3
# outcome[0:3]  
# outcome[50:53]  
# outcome[75:80]  

# # %% [markdown]
# # ##### Let us explore some outcome_decoded values

# # %%
# outcome_decoded = [x.argmax() for x in outcome]


# # %%
# outcome_decoded[0:3] #[1,1,1]
# outcome_decoded[50:53] #[2,2,0]
# outcome_decoded[75:80] # [0, 0, 0, 0]

# # %% [markdown]
# # ### Perform the PCA and t-SNE using scikit-learn

# # %%
# import sklearn.decomposition as sk_decomp
# tc1_library.run_and_plot_pca_and_tsne(TC1data3, outcome_decoded)

# # %% [markdown]
# # ## Create a binary dataset instead of a multi-class one
# # %% [markdown]
# # ### Current outcome distribution of the 150 samples

# # %%
# pd.value_counts(outcome_decoded)

# # %% [markdown]
# # ### Get the indexes of the data that correspond to classes 0 or 1 only (excluding class 2)

# # %%
# binary_indexes = np.where(np.array(outcome_decoded)!=2)[0]

# # %% [markdown]
# # ### Recreate the data structures of the same types that was used in the original analysis above, except this time with just two classes

# # %%
# TC1data2 = TC1data3.iloc[binary_indexes,:]
# outcome2 = outcome[binary_indexes,:]

# # %% [markdown]
# # ### Decode the new outcome matrix just like we did above, and print out the outcome distribution of the new set of samples

# # %%
# outcome_decoded2 = [x.argmax() for x in outcome2]
# pd.value_counts(outcome_decoded2)

# # %% [markdown]
# # ### Check our new dataset by performing the same unsupervised learning that we did above

# # %%
# tc1_library.run_and_plot_pca_and_tsne(TC1data2, outcome_decoded2)

# # %% [markdown]
# # ## Next Steps
# # %% [markdown]
# # * Pick up with the Jupyter notebook from the cell that splits the data into the training and test sets, but now replacing TC1data3 with TC1data2 and outcome with outcome2
# # * Work through the notebook until at least the model has been trained (the cell with history = model.fit(...)), resulting in a binary classifier
# # * Apply gene/feature importance tools to the resulting model in order to determine which genes best contribute to discriminating between cancer classes 0 and 1
# # %% [markdown]
# # ## Saving two-class datafiles to disk
# # ### Save the data and labels to CSV format in the repository's data directory

# # %%
# TC1data2.reset_index(drop=True).to_csv('Data/X_two_classes.csv')

# pd.Series(outcome_decoded2).to_csv('Data/y_two_classes.csv')

# # %% [markdown]
# # Test that we've exported the data correctly by reading them back in and running the unsupervised learning analyses

# # %%
# X = pd.read_csv('Data/X_two_classes.csv', index_col=0)

# y = pd.read_csv('Data/y_two_classes.csv', index_col=0, squeeze=True)

# tc1_library.run_and_plot_pca_and_tsne(X, y)


# # %%
# You are viewing the Jupyter Notebook from ML-TC1 GitHub repository, https://github.com/ravichas/ML-TC1 
