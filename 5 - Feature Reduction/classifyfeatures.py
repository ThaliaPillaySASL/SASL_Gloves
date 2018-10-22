# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 21:43:03 2018

@author: thali
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 20:17:27 2018

@author: thali
"""
import keras
from matplotlib import pyplot as plt
from IPython.display import clear_output
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import pandas as pd
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
dataframe = pandas.read_csv("all.csv", header=1)
dataframe.columns = ['Letter','F1', 'F2','F3','F4','F5','C1','C2','C3','C4','C5','O']
dataset = dataframe.values
X = dataset[:,1:12].astype(float)
Y = dataset[:,0]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()  
X = sc.fit_transform(X)  
dfx = pd.DataFrame(data=X,columns=dataframe.columns[1:])

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

#_______________________*** FEATURE EXTRACTION ***_______________________________
#pca http://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
from sklearn.decomposition import PCA
pca = PCA(n_components=None)  
dfx_pca = pca.fit(dfx)
dfx_trans = pca.transform(dfx)
dfx_trans = pd.DataFrame(data=dfx_trans)


#_______________________*** FEATURE EXTRACTION ***_______________________________

class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();
        
plot_losses = PlotLosses()
#_______________________*** FEATURE EXTRACTION ***_______________________________
# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(5, input_dim=11,init='normal', activation='relu'))
	model.add(Dense(24, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

model = baseline_model()
model.fit(dfx_trans,dummy_y, nb_epoch=50,callbacks=[plot_losses],batch_size=1,verbose=2) 
scores = model.evaluate(dfx_trans,dummy_y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
#results = cross_val_score(estimator, X, dummy_y, cv=kfold)
#print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

#model_json = model.to_json()
#with open("ModelFlex.json","w") as json_file:
#    json_file.write(model_json)
#model.save_weights("ModelFlex.h5")
#print ("Saved model to Disk")


