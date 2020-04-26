import os
import numpy as np
import pandas as pd

from scipy.stats import boxcox
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

from video_features import VideoFeatures
from audio_features import AudioFeatures

class Data:

	def __init__(self, config, options, pars):
		self.options = options
		self.config = config
		self.pars = pars
		self.video_features = config['video']['video_features']
		self.audio_features = config['audio']['audio_features']
		if self.video_features:
			self.video = VideoFeatures(config, options, pars)
		if self.audio_features:
			self.audio = AudioFeatures(config)

	# TODO: add test features and labels when available
	def load_data(self):
		"""
			Loads all data depending on parameters provided in the config file.
			Combine audio and video features if required

			Returns
			-------
			X_train, y_train, X_test, y_test : pd.DataFrame
				Final feature and label vectors ready for training and testing.
		"""

		if self.audio_features and self.video_features:
			video_data = self.load_video_features()
			audio_data = self.load_audio_features()
			X = self.combine_features(video_data, audio_data)
		elif self.video_features:
			X = self.load_video_features()
		elif self.audio_features:
			X = self.load_audio_features()

		# Split the labels according to indexes
		y = self.load_labels()
		
		X_train, y_train, X_test, y_test = self.split_data(X, y)
		X_train, X_test = self.preprocess(X_train, X_test)
	
		return X_train, y_train, X_test, y_test

	def preprocess(self, X_train, X_test):
		"""
			Scale and reduce dimensionality of input features

			Parameters:
			-----------
			X_train, X_test : ndarray (n, n_features)
			Arrays of input features where each row should be a single feature set

			Returns
			-------
			X_train, X_test : ndarray (n, n_in)
				Scaled and reduced features
		"""
		train_shape = X_train.shape
		test_shape = X_test.shape
		"""
		# Used to add before boxcox transformation to ensure all values are positive
		sv = 1
		X_train, maxlog = boxcox(X_train.flatten() + sv)
		X_train = X_train.reshape(train_shape)
		X_test = boxcox(X_test.flatten() + sv, maxlog).reshape(test_shape)
		"""
		scaler = MinMaxScaler().fit(X_train)
		X_train = scaler.transform(X_train)
		X_test = scaler.transform(X_test)
		
		pca = PCA().fit(X_train)
		n_components = np.where(np.cumsum(pca.explained_variance_ratio_) > self.pars['PCA']['combined_components'])[0][0]
		pca = PCA(n_components=n_components).fit(X_train)
		X_train = pca.transform(X_train)
		X_test = pca.transform(X_test)
		
		scaler = MinMaxScaler().fit(X_train)
		X_train = scaler.transform(X_train)
		X_test = scaler.transform(X_test)
		
		return X_train, X_test

	def load_labels(self):
		"""
			Loads all of the labels for all data parts (test, train and dev)

			Returns
			-------
			labels : pd.DataFrame
				A pandas dataframe of labels with index representing invividual patient
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
		video_data = list()
		video_data.append(self.video.get_video_data(data_part='Training'))
		video_data.append(self.video.get_video_data(data_part='Development'))
		if self.options.mode == 'test':
			video_data.append(self.video.get_video_data(data_part='Testing'))
		return self.prep_features(video_data)

	def load_audio_features(self):
		audio_train = self.audio.get_features('training')
		audio_dev = self.audio.get_features('development')
		audio_train.index = self.filename_to_index(audio_train.index)
		audio_dev.index = self.filename_to_index(audio_dev.index)
		audio_train = self.combine_tasks(audio_train)
		audio_dev = self.combine_tasks(audio_dev)
		return audio_train, audio_dev
	
	@staticmethod
	def split_data(X, y):
		if len(X) == 3:
			X_train = pd.concat([X[0], X[1]])
			X_test = X[2]
		else:
			X_train, X_test = X[0], X[1]
		
		y_train = y.loc[X_train.index]
		y_test = y.loc[X_test.index]
		
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
