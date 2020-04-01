import os
import pandas as pd

from video_features import VideoFeatures

class Data:

	def __init__(self, config):
		self.config = config
		self.video = VideoFeatures(config)
	
	# TODO: add test features and labels when availiable
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
		
		y = self.load_labels()
		y_train = y.loc[X_train.index.get_level_values(0)]
		y_dev = y.loc[X_dev.index.get_level_values(0)]
	
		return X_train, y_train, X_dev, y_dev
	
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
		video_train = self.video.get_videos(data_part='Training')
		video_dev = self.video.get_videos(data_part='Development')
		video_dev.index = self.filename_to_index(video_dev.index)
		return video_train, video_dev
	
	def load_audio_features(self):
		# TODO: load different audio featuers
		return None
	
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
