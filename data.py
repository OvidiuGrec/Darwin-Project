import os
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import boxcox
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from video_features import VideoFeatures
from audio_features import AudioFeatures


class Data:

	def __init__(self, config, options, pars):
		self.options = options
		self.config = config
		self.pars = pars
		self.feature_type = config['general']['feature_type']
		self.fusion = self.config['combined']['fusion']
		if self.feature_type in ['video', 'combined']:
			self.video = VideoFeatures(config, options, pars)
		if self.feature_type in ['audio', 'combined']:
			self.audio = AudioFeatures(config)
		if not self.video.fdhh:
			self.seq_length = pars['VanilaLSTM']['data']['seq_length']

	def load_data(self, feature_type):
		"""
			Loads all data depending on parameters provided in the config file.
			Combine audio and video features if required

			Returns
			-------
			X_train, y_train, X_test, y_test : pd.DataFrame
				Final feature and label vectors ready for training and testing.
		"""

		X = None
		y = self.load_labels()

		X_train, y_train, X_test, y_test = [np.array([]) for _ in range(4)]
		if feature_type == 'combined':
			features = {'video': self.load_video_features(), 'audio': self.load_audio_features()}
			if self.fusion == 'early':
				X = self.combine_features(features['video'], features['audio'])
			elif self.fusion == 'mid':
				for feature_type in ['video', 'audio']:
					_X_train, y_train, _X_test, y_test = self.split_data(features[feature_type], y)
					_X_train, _X_test = self.preprocess(feature_type, _X_train, _X_test)
					X_train = np.hstack((X_train, _X_train)) if X_train.size != 0 else _X_train
					X_test = np.hstack((X_test, _X_test)) if X_test.size != 0 else _X_test
		elif feature_type == 'video':
			X = self.load_video_features()
		elif feature_type == 'audio':
			X = self.load_audio_features()

		if X is not None:
			X_train, y_train, X_test, y_test = self.split_data(X, y)
			X_train, X_test = self.preprocess(feature_type, X_train, X_test)
		else:
			raise Exception("Invalid feature type has been provided (Should be from (audio, video, combined)")
		
		if not self.video.fdhh:
			y_train = y_train.iloc[::self.seq_length, :].copy()
			y_test = y_test.iloc[::self.seq_length, :].copy()
		
		idx_tr, idx_te = np.arange(X_train.shape[0]), np.arange(X_test.shape[0])
		np.random.shuffle(idx_tr), np.random.shuffle(idx_te)
		
		return X_train[idx_tr], y_train.iloc[idx_tr], X_test[idx_te], y_test.iloc[idx_te]

	def preprocess(self, feature_type, X_train, X_test):
		"""
			Scale and reduce dimensionality of input features

			Parameters:
			-----------
			X_train, X_test : np.array (n, n_features)
			Arrays of input features where each row should be a single feature set

			Returns
			-------
			X_train, X_test : np.array (n, n_in)
				Scaled and reduced features
		"""
		scaler = self.config[feature_type][f'{feature_type}_scaler'].split('+')
		scale_over = self.config[feature_type][f'{feature_type}_scale_over'].split('+')
		if len(scaler) > 1:
			scaler_idx = 1
		else:
			scaler_idx = 0
		
		if scaler[0] == 'boxcox':
			X_train, X_test = self.boxcox_transform(X_train, X_test)
		
		if scaler[scaler_idx] == 'minmax' or scaler[scaler_idx] == 'standard':
			X_train, X_test = self.scale(X_train, X_test, scale=scaler[scaler_idx], scale_over=scale_over[scaler_idx])
		elif self.options.verbose:
			print('No scaler has been used before PCA. If this behaviour is unintentional check configurations.')
			
		try:
			pca_pars = self.pars['PCA'][f'{feature_type}_components']
			X_train, X_test = self.pca_transform(X_train, X_test, pca_components=pca_pars)
		except KeyError:
			if self.options.verbose:
				print('No pca performed during preprocessing. If this behaviour is unintentional check parameters.')
				
		X_train, X_test = self.scale(X_train, X_test, scale='minmax')
		
		if feature_type == 'video' and not self.video.fdhh:
			# Reshape for LSTM:
			X_train = X_train.reshape(-1, self.seq_length, X_train.shape[-1])
			X_test = X_test.reshape(-1, self.seq_length, X_test.shape[-1])
			if self.options.verbose:
				print(f"Training input shape for the LSTM is {X_train.shape}")
			
		return X_train, X_test

	def load_labels(self):
		"""
			Loads all of the labels for all data parts (test, train and dev)

			Returns
			-------
			labels : pd.DataFrame
				A pandas dataframe of labels with index representing individual patient
				and corresponding to the value of there BDI-II score.
		"""
		folder = self.config['folders']['labels_folder']
		labels = pd.DataFrame()
		for (dirpath, _, filenames) in os.walk(folder):
			if filenames:
				for file in filenames:
					labels[file[:5]] = pd.read_csv(f'{dirpath}/{file}').columns.values

		return labels.transpose()

	def load_video_features(self):
		train_video = self.video.get_video_data(data_part='Training')
		dev_video = self.video.get_video_data(data_part='Development')
		video_data = [train_video,  dev_video]
		if self.options.mode == 'test':
			video_data.append(self.video.get_video_data(data_part='Testing'))
		if self.video.fdhh:
			video_data = self.prep_features(video_data)
		return video_data

	def load_audio_features(self):
		audio_data = list()
		audio_data.append(self.audio.get_features('training'))
		audio_data.append(self.audio.get_features('development'))
		if self.options.mode == 'test':
			audio_data.append(self.audio.get_features('testing'))
		return self.prep_features(audio_data)
	
	@staticmethod
	def split_data(X, y):
		if len(X) == 3:
			X_train = pd.concat([X[0], X[1]])
			X_test = X[2]
		else:
			X_train, X_test = X[0], X[1]
		
		y_train = y.loc[X_train.index.get_level_values(0)]
		y_test = y.loc[X_test.index.get_level_values(0)]
		
		return X_train.values, y_train, X_test.values, y_test
		
	def prep_features(self, data):
		for i, part in enumerate(data):
			part.index = self.filename_to_index(part.index)
			data[i] = self.combine_tasks(part)
		return data

	@staticmethod
	def combine_features(video, audio):
		# audio.index = video.index
		comb_data = []
		for i in range(len(video)):
			comb_data.append(pd.concat([video[i], audio[i]], axis=1))
		return comb_data

	@staticmethod
	def combine_tasks(data):
		"""
			Combines data from same tasks
			----------
			data: pd.DataFrame
				Pandas data frame with index | Patient # | Task | and features in columns

			Returns
			-------
			combined : pd.DataFrame
				A pandas data frame with tasks combined into single vector and indexed by | Patient # |
		"""
		idx = pd.IndexSlice
		freeform = data.loc[idx[:, 'Freeform'], :].sort_index().droplevel(1)
		northwin = data.loc[idx[:, 'Northwin'], :].sort_index().droplevel(1)
		assert (freeform.index == northwin.index).all(), 'Indexes in two tasks not equal'
		
		n_columns = freeform.shape[1]
		northwin.columns = np.arange(n_columns, n_columns+n_columns)
		combined = pd.concat([freeform, northwin], axis=1)
		
		return combined
		
	@staticmethod
	def filename_to_index(index):
		"""
			Splits filename index into a MultiIndex with patient number and task
			Parameters
			----------
			index: pd.Index
				Index in format (patient_rep_task)

			Returns
			-------
			new_index : pd.MultiIndex
				A pandas MultiIndex corresponding to (patient_rep, task)
		"""
		new_index = [(f'{n[0]}_{n[1]}', n[2]) for n in [x.split('_') for x in index]]
		return pd.MultiIndex.from_tuples(new_index)
	
	def boxcox_transform(self, X_train, X_test):
		# TODO: add feature-wise boxcox transformation
		train_shape = X_train.shape
		test_shape = X_test.shape
		
		if self.options.verbose:
			print('Performing boxcox transformation')
			fig, (ax1, ax2) = plt.subplots(1, 2)
			fig.suptitle('Comparing distributions before (left) and after (right) boxcox transformation')
			ax1.hist(X_train.flatten(), label='train')
			ax1.hist(X_test.flatten(), label='test')
			ax1.legend()
		
		# Used to add before boxcox transformation to ensure all values are positive
		sv = 1
		X_train, maxlog = boxcox(X_train.flatten() + sv)
		X_train = X_train.reshape(train_shape)
		X_test = boxcox(X_test.flatten() + sv, maxlog).reshape(test_shape)
		
		if self.options.verbose:
			ax2.hist(X_train.flatten(), label='train')
			ax2.hist(X_test.flatten(), label='test')
			ax2.legend()
		
		plt.show()
			
		return X_train, X_test
	
	@staticmethod
	def scale(X_train, X_test, scale='minmax', scale_over='feature'):
		# TODO: add visualization
		if scale == 'minmax':
			if scale_over == 'feature':
				min_ = np.min(X_train, axis=0)
				max_ = np.max(X_train, axis=0)
			elif scale_over == 'full':
				min_ = np.min(X_train)
				max_ = np.max(X_test)
			
			diff = max_ - min_
			diff[diff == 0] = 1
			X_train = (X_train - min_) / diff
			X_test = (X_test - min_) / diff
			
		elif scale == 'standard':
			if scale_over == 'feature':
				mean = np.mean(X_train, axis=0)
				std = np.std(X_train, axis=0)
			else:
				mean = np.mean(X_train)
				std = np.std(X_train)
			
			X_train = (X_train - mean) / std
			X_test = (X_test - mean) / std
		
		return X_train, X_test
	
	@staticmethod
	def pca_transform(X_train, X_test, pca_components=0.9):
		# TODO: add visualization
		if pca_components < 1:
			pca = PCA().fit(X_train)
			pca_components = np.where(np.cumsum(pca.explained_variance_ratio_) > pca_components)[0][0]
			mlflow.log_param('n_features', pca_components)
		pca = PCA(n_components=pca_components).fit(X_train)
		return pca.transform(X_train), pca.transform(X_test)
