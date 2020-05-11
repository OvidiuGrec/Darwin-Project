import numpy as np
import pandas as pd

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor, VotingRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, PReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import L1L2


class DepressionModel:
	
	def __init__(self, feature_type, config, input_shape, pars=None):
		self.in_shape = input_shape
		self.config = config
		self.pars = pars
		self.model_names = config[f'{feature_type}_model'].split('+')
		self.model_weights = [float(v) for v in config[f'{feature_type}_model_weights'].split('+')]
		self.models = self.get_models()
	
	def get_models(self):
		
		model_switcher = {
			'PLS': self.PLS(),
			'LR': self.LR(),
			'Ridge': self.Ridge(),
			'DecisionTree': self.DecisionTree(),
			'SVR': self.SVR(),
			'ElasticNet': self.ElasticNet(),
			'GradientBoosting': self.GradientBoosting(),
			'AdaBoost': self.AdaBoost(),
		    'RandomForest': self.RandomForest(),
			'VotingReg': self.VotingReg(),
			'FNN': self.FNN(),
			'VanilaLSTM': self.VanilaLSTM()
		}
		
		models = []
		for model_n in self.model_names:
			models.append(model_switcher[model_n])
		
		del model_switcher
		
		return models
	
	def train(self, X, y):
		for i in range(len(self.models)):
			model_name = self.model_names[i]
			model = self.models[i]
			model.fit(X, y.values.flatten(), **self.pars[model_name]['train'])
	
	def predict(self, X):
		n_models = len(self.models)
		model_w = self.model_weights
		final_pred = 0
		for i in range(n_models):
			model = self.models[i]
			w = model_w[i]
			pred = w * model.predict(X)
			if i == 0:
				final_pred = pred.flatten()
			else:
				final_pred += pred.flatten()
		
		return np.max((final_pred, np.zeros(final_pred.size)), axis=0)

	def PLS(self):  # Partial Least Squares
		try:
			pls2 = PLSRegression(**self.pars['PLS']['model'])
		except KeyError:
			pls2 = None
		return pls2
	
	def LR(self):  # LinearRegression
		try:
			reg = LinearRegression(**self.pars['LR']['model'])
		except KeyError:
			reg = None
		return reg

	def Ridge(self):
		try:
			reg = Ridge(**self.pars['Ridge']['model'])
		except KeyError:
			reg = None
		return reg

	def DecisionTree(self):
		try:
			model = DecisionTreeRegressor(**self.pars['DecisionTree']['model'])
		except KeyError:
			model = None
		return model

	def SVR(self):
		try:
			model = SVR(**self.pars['SVR']['model'])
		except KeyError:
			model = None
		return model

	def GradientBoosting(self):
		try:
			model = GradientBoostingRegressor(**self.pars['GradientBoosting']['model'])
		except KeyError:
			model = None
		return model

	def ElasticNet(self):
		try:
			model = ElasticNet(**self.pars['ElasticNet']['model'])
		except KeyError:
			model = None
		return model
	
	def AdaBoost(self):
		try:
			model = AdaBoostRegressor(**self.pars['AdaBoost']['model'])
		except KeyError:
			model = None
		return model
	
	def RandomForest(self):
		try:
			model = RandomForestRegressor(**self.pars['RandomForest']['model'])
		except KeyError:
			model = None
		return model
	
	def VotingReg(self):
		try:
			model = VotingRegressor([('svr', self.SVR()), ('lr', self.LR()), ('el', self.ElasticNet())])
		except KeyError:
			model = None
		return model
	
	def FNN(self):  # FeedForward Neural Network
		try:
			pars = self.pars['FNN']['model']
			ker_reg = L1L2(l1=pars['ker_reg1'], l2=pars['ker_reg2'])
			model = Sequential([
				Dense(pars['l1'], input_dim=self.in_shape[1], kernel_regularizer=ker_reg),
				PReLU(),
				Dropout(pars['d1']),
				Dense(pars['l2'], kernel_regularizer=ker_reg),
				PReLU(),
				Dropout(pars['d2']),
				Dense(1, activation='linear')
			])
			model.compile(loss='mae', optimizer=Adam(lr=pars['lr']))
			self.pars['FNN']['train']['callbacks'] = [EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10)]
			
		except KeyError:
			model = None
		return model
	
	def VanilaLSTM(self):  # Long-Short-Term-Memory Network
		try:
			pars = self.pars['VanilaLSTM']['model']
			ker_reg = L1L2(l1=pars['ker_reg1'], l2=pars['ker_reg2'])
			rec_reg = L1L2(l1=pars['rec_reg1'], l2=pars['rec_reg2'])
			model = Sequential()
			model.add(LSTM(pars['l1'], input_shape=(self.in_shape[1], self.in_shape[2]), activation='relu',
			               recurrent_regularizer=rec_reg, kernel_regularizer=ker_reg))
			model.add(Dropout(pars['d1']))
			model.add(Dense(pars['l2'], activation='relu'))
			model.add(Dropout(pars['d2']))
			model.add(Dense(1, activation='linear'))
			
			model.compile(loss='mean_squared_error', optimizer=Adam(lr=pars['lr']))
			self.pars['VanilaLSTM']['train']['callbacks'] = [EarlyStopping(monitor='val_loss', min_delta=0.001, patience=100)]
		except KeyError:
			model = None
		return model