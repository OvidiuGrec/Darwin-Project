import os
import numpy as np
import pandas as pd
import numpy as np

from scipy.stats import boxcox
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

from video_features import VideoFeatures
from audio_features import AudioFeatures

class Data:

	def __init__(self, config):
		self.config = config
		if config['video_features']:
			self.video = VideoFeatures(config)
		if config['audio_features']:
			self.audio = AudioFeatures(config)

	# TODO: add test features and labels when available
	def load_data(self):
		"""
			Loads all data depending on parameters provided in the config file.
			Combine audio and video features if required

			Returns
			-------
			X_train, y_train, X_dev, y_dev : pd.DataFrame
				Final feature and label vectors ready for training and testing.
		"""
		video_features = self.config['video_features']
		audio_features = self.config['audio_features']

		if audio_features and video_features:
			video_train, video_dev = self.load_video_features()
			audio_train, audio_dev = self.load_audio_features()
			# TODO: Combine audio and video features
			# X_train = self.combine_features(video_train, audio_train)
			# X_dev = self.combine_features(video_dev, audio_dev)
		elif video_features:
			X_train, X_dev = self.load_video_features()
		elif audio_features:
			X_train, X_dev = self.load_audio_features()

		# Split the labels according to indexes
		y = self.load_labels()
		y_train = y.loc[X_train.index.get_level_values(0)]
		y_dev = y.loc[X_dev.index.get_level_values(0)]
		
		X_train, X_dev = self.preprocess(X_train.values, X_dev.values)
	
		return X_train, y_train, X_dev, y_dev

	def preprocess(self, X_train, X_dev):
		"""
			Scale and reduce dimensionality of input features

			Parameters:
			-----------
			X_train, X_dev : ndarray (n, n_features)
			Arrays of input features where each row should be a single feature set

			Returns
			-------
			X_train, X_dev : ndarray (n, n_in)
				Scaled and reduced features
		"""
		train_shape = X_train.shape
		dev_shape = X_dev.shape
		# Used to add before boxcox transformation to ensure all values are positive
		sv = -np.min(np.vstack((X_train, X_dev))) + 0.0000001

		X_train, maxlog = boxcox(X_train.flatten() + sv)
		X_train = X_train.reshape(train_shape)
		X_dev = boxcox(X_dev.flatten() + sv, maxlog).reshape(dev_shape)
		
		scaler = StandardScaler().fit(X_train)
		X_train = scaler.transform(X_train)
		X_dev = scaler.transform(X_dev)
		
		pca = PCA().fit(X_train)
		n_components = np.where(np.cumsum(pca.explained_variance_ratio_) > self.config['var_ratio'])[0][3]
		pca = PCA(n_components=n_components).fit(X_train)
		X_train = pca.transform(X_train)
		X_dev = pca.transform(X_dev)
		
		scaler = MinMaxScaler().fit(X_train)
		X_train = scaler.transform(X_train)
		X_dev = scaler.transform(X_dev)
		
		return X_train, X_dev

	def load_labels(self):
		"""
			Loads all of the labels for all data parts (test, train and dev)

			Returns
			-------
			labels : pd.DataFrame
				A pandas dataframe of labels with index representing invividual patient
				and corresponding to the value of there BDI-II score.
		"""
		folder = self.config['labels_folder']
		labels = pd.DataFrame()
		for (dirpath, _, filenames) in os.walk(folder):
			if filenames:
				for file in filenames:
					labels[file[:5]] = pd.read_csv(f'{dirpath}/{file}').columns.values

		return labels.transpose()

	def load_video_features(self):
		video_train = self.video.get_video_data(data_part='Training')
		video_dev = self.video.get_video_data(data_part='Development')
		video_train.index = self.filename_to_index(video_train.index)
		video_dev.index = self.filename_to_index(video_dev.index)
		video_train = self.combine_tasks(video_train)
		video_dev = self.combine_tasks(video_dev)
		return video_train, video_dev

	def load_audio_features(self):
		audio_train = self.audio.get_features('training')
		audio_dev = self.audio.get_features('development')
		audio_train.index = self.filename_to_index(audio_train.index)
		audio_dev.index = self.filename_to_index(audio_dev.index)
		audio_train = self.combine_tasks(audio_train)
		audio_dev = self.combine_tasks(audio_dev)
		return audio_train, audio_dev

	# @staticmethod
	# def combine_features():

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
