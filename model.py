import numpy as np

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

import keras
from keras.models import Sequential
from keras.layers import Dense


class DepressionModel:
	
	def __init__(self, config, pars=None):
		self.config = config
		self.pars = pars
		self.model_types = config['model_type'].split('+')
		self.model_names = config['model_name'].split('+')
		self.model_weights = [float(v) for v in config['model_weights'].split('+')]
		self.models = self.get_models()
	
	def get_models(self):
		
		model_switcher = {
			'PLS': self.PLS(),
			'LR': self.LR(),
			'FNN': self.FNN()
		}
		
		models = []
		for model_n in self.model_names:
			models.append(model_switcher[model_n])
		
		return models
	
	def train(self, X, y):
		for i in range(len(self.models)):
			model_type = self.model_types[i]
			model_name = self.model_names[i]
			model = self.models[i]
			
			if model_type == 'sklearn':
				model.fit(X, y)
			elif model_type == 'keras':
				model.fit(X, y, **self.pars[model_name]['train'])
	
	def predict(self, X):
		n_models = len(self.models)
		model_w = self.model_weights
		for i in range(n_models):
			model = self.models[i]
			w = model_w[i]
			pred = w * model.predict(X)
			if i == 0:
				final_pred = pred
			else:
				final_pred += pred
		
		return final_pred
	
	def validate_model(self, X, y):
		
		pred = self.predict(X)

		mae = mean_absolute_error(y, pred)
		mse = mean_squared_error(y, pred)
		rmse = np.sqrt(mse)
		
		return mae, rmse
		
	def PLS(self):  # Partial Least Squares
		try:
			pls2 = PLSRegression(**self.pars['PLS'])
		except KeyError:
			pls2 = None
		return pls2
	
	def LR(self):  # LinearRegression
		try:
			reg = LinearRegression(**self.pars['LR'])
		except KeyError:
			reg = None
		return reg
	
	def FNN(self):  # FeedForward Neural Network
		try:
			pars = self.pars['FNN']['model']
			model = Sequential()
			model.add(Dense(pars['l1'], input_dim=self.config['n_in'], activation='relu'))
			model.add(Dense(pars['l2'], activation='relu'))
			model.add(Dense(1, activation='linear'))
			
			model.compile(loss='mean_squared_error', optimizer='adam',
			              metrics=['mean_squared_error', 'mean_absolute_error'])
		except KeyError:
			model = None
		return model