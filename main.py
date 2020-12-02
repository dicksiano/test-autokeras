from config import Constants
from input_reader import read_data
from neural_network import NeuralNetwork
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.callbacks import ModelCheckpoint

from keras import backend as K
import itertools
import numpy as np
import tensorflow as tf


from sklearn.preprocessing import Normalizer, MinMaxScaler
#from autokeras import StructuredDataRegressor

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(hist['epoch'], hist['mean_absolute_error'], label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label = 'Val Error')
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'], label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'], label = 'Val Error')
  plt.legend()
  plt.show()

def dataset():
	with open('meta_cp0.pkl', 'rb') as pkl:
		X_train_cp0, y_train_cp0 = pickle.load(pkl)
	with open('meta_cpRnd.pkl', 'rb') as pkl:
		X_train_cpRnd, y_train_cpRnd = pickle.load(pkl)

	print(len(X_train_cp0), len(X_train_cp0[0]), len(X_train_cpRnd), len(X_train_cpRnd[0]))
	
	#print(np.max(y_train_cp0), np.min(y_train_cp0), np.mean(y_train_cp0))
	#print(np.max(y_test_cp0), np.min(y_test_cp0), np.mean(y_test_cp0))
	#
	#print(np.max(y_train_cpRnd), np.min(y_train_cpRnd), np.mean(y_train_cpRnd))
	#print(np.max(y_test_cpRnd), np.min(y_test_cpRnd), np.mean(y_test_cpRnd))

	assert(len(X_train_cp0) == len(y_train_cp0))
	#assert(len(X_test_cp0) == len(y_test_cp0))
	assert(len(X_train_cpRnd) == len(y_train_cpRnd))
	#assert(len(X_test_cpRnd) == len(y_test_cpRnd))

	X_train = list(itertools.chain(X_train_cp0, X_train_cpRnd))
	y_train = list(itertools.chain(y_train_cp0, y_train_cpRnd))
	#X_test  = list(itertools.chain(X_test_cp0, X_test_cpRnd)) #X_test_cp0 
	#y_test  = list(itertools.chain(y_test_cp0, y_test_cpRnd)) #y_test_cp0
	print(len(X_train_cp0), len(X_train_cpRnd))

	assert( len(X_train) == len(y_train) )
	assert( len(X_train) == (len(X_train_cp0)+len(X_train_cpRnd)) )
	#assert( len(X_test) == len(y_test) )
	#assert( len(X_test) == (len(X_test_cp0)+len(X_test_cpRnd)) )

	return np.array(X_train).astype(np.float), np.array(y_train).astype(np.float), np.array(X_train_cp0).astype(np.float), np.array(y_train_cp0).astype(np.float), 	np.array(X_train_cpRnd).astype(np.float), np.array(y_train_cpRnd).astype(np.float)

def main():

	# Setup dataset
	X_train, y_train, X_train_cp0, y_train_cp0, X_train_cpRnd, y_train_cpRnd = dataset()
	
	
	#sc = MinMaxScaler()
	#X_train = sc.fit_transform(X_train)
	#n = Normalizer()
	#X_train = n.fit_transform(X_train)
	print(X_train[0])
	#mean = X_train.mean(axis=0)
	#std = X_train.std(axis=0)
	#X_train = (X_train - mean) / std
	


	# define the search
	#search = StructuredDataRegressor(max_trials=15, loss='mean_absolute_error')
	# perform the search
	#search.fit(x=X_train, y=y_train, verbose=1)
	# evaluate the model
	#mae, _ = search.evaluate(X_train, y_train, verbose=0)





	nn = NeuralNetwork(Constants.hyperparameters, X_train_cp0, y_train_cp0)
	#K.set_session(tf.get_default_session(), '20201120-052235_0/model')
	print( nn.show_configuration() + "_" + str(X_train.shape[0] ) )
	#print(np.max(y_train), np.min(y_train), np.mean(y_train))
	#print(np.max(y_test), np.min(y_test), np.mean(y_test))

	history = nn.train(X_train_cp0, y_train_cp0, X_train_cp0, y_train_cp0)   #, X_test, y_test)
	print(nn.model.metrics_names, nn.evaluate_model(X_train_cp0, y_train_cp0))
	#print(nn.model.metrics_names, nn.evaluate_model(X_test_cp0, y_test_cp0))
	
	print(nn.model.metrics_names, nn.evaluate_model(X_train_cpRnd, y_train_cpRnd))
	#print(nn.model.metrics_names, nn.evaluate_model(X_test_cpRnd, y_test_cpRnd))

	#plot_history(history)
	nn.model.save(nn.show_configuration() + "_" + str(X_train.shape[0]) + "___" + str(len(history.epoch)) )

if __name__ == '__main__':
	main()
