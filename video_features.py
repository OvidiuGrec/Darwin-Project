import os
import gc
import cv2
import progressbar
import numpy as np
import pandas as pd
from sys import platform

import dask.dataframe as dd
from dask_ml.preprocessing import MinMaxScaler

from mtcnn import MTCNN
from keras import Model
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

from helper import save_to_file, load_from_file
from preprocess import pca_transform


class VideoFeatures:

	def __init__(self, config, options, pars):
		self.input_size = (224, 224, 3)
		self.folders = config['folders']
		self.options = options
		self.pars = pars
		self.vgg_v, self.vgg_l, fdhh = config['general']['video_features'].split('_')
		self.fdhh = bool(fdhh)
		self.feature_folder = f'{self.folders["video_folder"]}/{self.vgg_v}_{self.vgg_l}'
		if not self.fdhh:
			try:
				self.seq_length = pars[config['model']['video_model']]['data']['seq_length']
			except KeyError:
				raise Exception('No sequence length provided from LSTM model under data section in parameters.')

	def get_video_data(self):
		"""
			Returns video feature data depending on parameter provided in config file.
			Performs fdhh algorithm if required otherwise return raw video (WARNING: Potential RAM overflow)

			Returns
			-------
			X_train, X_test
			
		"""
		if self.options.mode == 'test':
			fdhh_path = (f'{self.feature_folder}_FD', f'train_test_fdhh.pic')
		else:
			fdhh_path = (f'{self.feature_folder}_FD', f'train_dev_fdhh.pic')
			
		# Return fdhh data if exists:
		if self.fdhh and not self.options.save_fdhh:
			if os.path.exists(f'{fdhh_path[0]}/{fdhh_path[1]}'):
				return load_from_file(f'{fdhh_path[0]}/{fdhh_path[1]}')
			
		X_train, X_test = self.get_train_test()
		
		if self.options.verbose:
			print('Scaling video features...')
		scaler = MinMaxScaler().fit(X_train)
		X_train = scaler.transform(X_train)
		X_test = scaler.transform(X_test)
		
		if self.fdhh:
			if self.options.verbose:
				print('Performing FDHH over train and test set... \n')
			meta = {i: 'f8' for i in range(X_train.shape[1]*self.pars['FDHH']['pattern_len'])}
			X_train = X_train.groupby('file').apply(self.FDHH, meta=meta).compute()
			X_test = X_test.groupby('file').apply(self.FDHH, meta=meta).compute()
			if self.options.save_fdhh:
				save_to_file(fdhh_path[0], fdhh_path[1], (X_train, X_test))
		else:
			if self.options.verbose:
				print('Reducing dimensionality using PCA... \n')
			n_components = self.pars['PCA']['per_frame_components']
			X_train, X_test, _ = pca_transform(X_train.compute(), X_test.compute(), n_components, use_pandas=True)
			X_train = self.split_videos(X_train)
			X_test = self.split_videos(X_test)
			
		return X_train, X_test
	
	def get_train_test(self):
		
		if self.options.verbose:
			print(f'Putting together video data... \n')
		
		data_parts = ['Training', 'Development']
		if self.options.mode == 'test':
			data_parts.append('Testing')
		all_data = []
		
		for data_part in data_parts:
			
			data_path = f'{self.feature_folder}/{data_part}'
			if not os.path.exists(data_path):
				self.encode_videos()
				
			files = os.listdir(data_path)
			
			# Pre-allocate memory for the DataFrame:
			idx = []
			for file in files:
				size, n_features = load_from_file(f'{data_path}/{file}').shape
				idx += list(np.repeat(file[:-4], size))
			all_videos = pd.DataFrame(data=np.empty((len(idx), n_features)), index=idx,
			                          columns=[f'f{i}' for i in np.arange(n_features)])
			all_videos.index = all_videos.index.rename('file')
			# Extract videos and put into DataFrame:
			for file in files:
				all_videos.loc[file[:-4]] = load_from_file(f'{data_path}/{file}')
			all_videos = dd.from_pandas(all_videos, npartitions=8)
			all_data.append(all_videos)
			del all_videos
		
		if self.options.mode == 'test':
			X_train, X_test = dd.concat([all_data[0], all_data[1]]), all_data[2]
		else:
			X_train, X_test = tuple(all_data)
		del all_data
		return X_train, X_test
	
	def encode_videos(self):
		"""
			Extracts vgg encoding from each of the frames in each video and stores those in a separate folder.
			Only need to be ran once for each vgg model.
		"""
		
		self.face_detector = MTCNN()
		self.encoder = self.get_vggface()
		
		folder = self.folders['raw_video_folder']
		
		for (dirpath, _, filenames) in os.walk(folder):
			if platform == 'linux' or platform == 'linux2' or platform == 'darwin':
				# linux and OSX
				split_path = dirpath.split('/')
			else:
				# windows
				split_path = dirpath.split('\\')
			if filenames:
				if self.options.verbose:
					print(f'Extracting features from {dirpath}')
				for file in progressbar.progressbar(filenames):
					encode_path = (f'{self.feature_folder}/{split_path[-2]}', f'{file[:14]}.pic')
					coord_path = (f'{self.folders["facial_data"]}', f'{file[:14]}.pic')
					if file.endswith('.mp4') and not os.path.exists(f'{encode_path[0]}/{encode_path[1]}'):
						faces, coords = self.video_faces(f'{dirpath}/{file}', f'{coord_path[0]}/{coord_path[1]}')
						encoding = self.vggface_encoding(faces)
						save_to_file(coord_path[0], coord_path[1], coords)
						save_to_file(encode_path[0], encode_path[1], encoding.reshape(encoding.shape[0], -1))
						del faces, encoding
						gc.collect()
						
	def video_faces(self, video_path, coord_path):
		"""
			 Extracts faces from a video returning array of rgb images of faces

			 Parameters
			 ----------
			 video_path : str
				 A path to the video file
				 
			 coord_path : str
			     Folder location and file name of the file with face coordinates
			     
			 Returns
			 -------
			 faces : ndarray (? * 4)
				 An array of faces corresponding to each frame in the video. Leaves nan values if a face is missing
		 """

		cap = cv2.VideoCapture(video_path)

		# Check if camera opened successfully
		if not cap.isOpened():
			print("Error opening video stream or file")
			return None
		
		n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		coords_present = os.path.exists(coord_path)
			
		if coords_present:
			all_coords = load_from_file(coord_path)
		else:
			all_coords = np.empty(shape=(n_frames, 4), dtype=np.int64)

		faces = np.empty(shape=(n_frames, self.input_size[0], self.input_size[1], self.input_size[2]))
		faces[:] = np.nan
		
		n_nan = 0
		i = -1
		while cap.isOpened():
			ret, frame = cap.read()
			if ret:
				i += 1
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				if not coords_present:
					all_coords[i] = self.get_face_coords(frame)
				c = all_coords[i]
				if (c == -1).all():
					n_nan += 1
					continue
				else:
					face = frame[c[0]:c[1], c[2]:c[3]]
				faces[i] = cv2.resize(face, (self.input_size[0], self.input_size[1])).astype('float64')
			else:
				break
		cap.release()
		
		if n_nan > 0:
			faces = self.fill_nan(faces)
			if self.options.verbose:
				print(f'Total number of NaN frames before interpolation in {video_path} is {n_nan}')
				print(f'Reduced to {np.count_nonzero(np.isnan(faces))/(224*224*3)}')
			
		return faces, all_coords
	
	def get_face_coords(self, frame):
		"""
			Detects and extracts a face from the image

			Parameters
			----------
			frame : ndarray (? * ? * 3)
				An RGB array of image
				
			Returns
			-------
			face : 4-tuple
				A face bounding box extracted from the original image and fitted for input size
		"""

		try:
			results = self.face_detector.detect_faces(frame)  # Detects faces in the image

			x1, y1, width, height = results[0]['box']  # Bounding box of first face
			x1, y1 = abs(x1), abs(y1)  # bug fix...
			x2, y2 = x1 + width, y1 + height
		except:
			x1, x2, y1, y2 = -1, -1, -1, -1
		return y1, y2, x1, x2
	
	def vggface_encoding(self, faces):
		"""
			Encodes an image of a face into a lower dimensional representation
			Parameters
			----------
			faces : ndarray (n * 224 * 224 * 3)
				An array of face images to be encoded
				
			encoder : object
				vgg_model for feature extraction
				
			Returns
			-------
			encoding : ndarray (n * X)
				Feature-vector of a face representations
		"""

		inputs = preprocess_input(faces, version=2)  # TODO: 1 for VGG and 2 for others
		yhat = self.encoder.predict(inputs)
		return yhat

	def get_vggface(self):
		"""
			Initializes a VGGFace convolutional neural net based on parameters provided in config file.
			See: https://github.com/rcmalli/keras-vggface/blob/master/keras_vggface/models.py
			Returns
			-------
			vgg_model_custom_layer : keras.model
				A keras CNN model with last layer representing features to be extracted.
		"""
		layers_select = {
			'32': 'fc6',
			'34': 'fc7',
			'35': 'fc7/relu',
			'AVGPOOL': 'avg_pool'  # resnet and senet
		}

		model_select = {
			'VGG': 'vgg16',
			'RES': 'resnet50',
			'SE': 'senet50'
		}

		# this returns layer-specific features:
		wanted_layer = layers_select[self.vgg_l]
		# TODO: change include top depending on the architecture
		vgg_model = VGGFace(model=model_select[self.vgg_v], input_shape=(224, 224, 3), include_top=False)

		out = vgg_model.get_layer(wanted_layer).output
		vgg_model_custom_layer = Model(inputs=vgg_model.input, outputs=out)

		return vgg_model_custom_layer

	def FDHH(self, video):
		"""
			A feature dynamic histogram encoding from paper by Jan, Asim:
			https://www.researchgate.net/publication/318798243_Artificial_Intelligent_System_for_Automatic_Depression_Level_Analysis_through_Visual_and_Vocal_Expressions
			
			----------
			Parameters
			----------
			video : ndarray (N * C)
				A video to be encoded as a feature dynamic histogram where N is number of frames and C is number of features.
			
			Returns
			-------
			encoding : fdhh (M * C)
				Feature vector of consequtive change in pixel values for M number of changes for each pixel C.
		"""
		video = video.values
		
		fdhh_pars = self.pars['FDHH']
		
		frames, components = video.shape  # (N, C) in the paper
		pattern_len = fdhh_pars['pattern_len']  # (M) in the paper
		thresh = fdhh_pars['thresh']  # (T) in the paper

		dynamics = np.abs(video[1:] - video[:-1])
		binary_d = np.where(dynamics > thresh, 1, 0).T  # (D(c,n)) in the paper

		fdhh = np.zeros((components, pattern_len))

		for c in range(components):
			count = 0
			for n in range(frames - 2):
				if binary_d[c, n + 1]:
					count += 1
				elif 0 < count <= pattern_len:
					fdhh[c, count - 1] += 1
					count = 0
				elif count > pattern_len:
					fdhh[c, pattern_len - 1] += 1
					count = 0
		
		return pd.Series(fdhh.flatten())
	
	def split_videos(self, video_data):
		files_idx = video_data.index.get_level_values(0)
		frame_idx = np.hstack([np.arange(len(np.where(files_idx == f))) for f in files_idx.unique()])
		video_data.index = pd.MultiIndex.from_tuples(zip(files_idx, frame_idx))
		new_idx = []
		for file in files_idx.unique():
			frames = frame_idx[np.where(files_idx == file)]
			cut_b = frames[-1] % self.seq_length
			frames = frames[cut_b+1:]
			new_idx += list(zip(np.repeat(file, len(frames)), frames.values))
		return video_data.loc[new_idx]
	
	@staticmethod
	def fill_nan(faces):
		init_shape = faces.shape
		faces = pd.DataFrame(faces.reshape(init_shape[0], -1)).interpolate(axis=0, limit_direction='both')
		faces = faces.values.reshape(init_shape)
		return faces
