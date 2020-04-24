import os
import cv2
import numpy as np
import pandas as pd
from sys import platform

from sklearn.preprocessing import MinMaxScaler
from helper import save_to_file, load_from_file

from mtcnn import MTCNN
from keras import Model
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input


class VideoFeatures:

	def __init__(self, config):
		self.input_size = (224, 224, 3)
		self.config = config
		self.vgg_v, self.vgg_l, fdhh = self.config['video_features'].split('_')
		self.fdhh = bool(fdhh)
		self.feature_folder = f'{self.config["video_folder"]}/{self.vgg_v}_{self.vgg_l}'

	def get_video_data(self, data_part):
		"""
			Returns video feature data depending on parameter provided in config file.
			Performs fdhh algorithm if required otherwise return raw video (WARNING: Potential RAM overflow)

			Parameters
			----------
			data_part : str
				A folder from which to extract data from. Should be either Training/Development/Testing

			Returns
			-------
			faces : ndarray (? * ? * ? * 3)
				An array of faces corresponding to each frame in the video. Leaves nan values if a face is missing
		"""

		data_path = f'{self.feature_folder}/{data_part}'
		scaler_path = (self.config['models_folder'], f'{self.vgg_v}_{self.vgg_l}_scaler')
		fdhh_path = (f'{self.feature_folder}_FD', f'{data_part}.pic')

		if self.fdhh and os.path.exists(f'{fdhh_path[0]}/{fdhh_path[1]}'):
			return load_from_file(f'{fdhh_path[0]}/{fdhh_path[1]}')

		if not os.path.exists(data_path):
			self.encode_videos()

		if data_part == 'Training':
			scaler = self.video_min_max(data_path)
			save_to_file(scaler_path[0], scaler_path[1], scaler)
		elif data_part == 'Development' and self.config['mode'] == 'test':
			scaler = load_from_file(f'{scaler_path[0]}/{scaler_path[1]}')
			scaler = self.video_min_max(data_path, scaler)
			save_to_file(scaler_path[0], scaler_path[1], scaler)
		else:
			scaler = load_from_file(f'{scaler_path[0]}/{scaler_path[1]}')

		files = os.listdir(data_path)
		if self.fdhh:
			fdhh_data = pd.DataFrame()
			for file in files:
				video_data = scaler.transform(load_from_file(f'{data_path}/{file}'))
				fdhh_data[file[:-4]] = self.FDHH(video_data).flatten()
			fdhh_data = fdhh_data.transpose()
			save_to_file(fdhh_path[0], fdhh_path[1], fdhh_data)
			return fdhh_data

		else:
			# TODO: Aggregate all raw video data (Maybe use Tensorflow generator?)
			return None

	def encode_videos(self):
		"""
			Extracts vgg encoding from each of the frames in each video and stores those in a separate folder.
			Only need to be ran once for each vgg model.
		"""
		
		face_detector = MTCNN()
		encoder = self.get_vggface()
		
		folder = self.config['raw_video_folder']

		for (dirpath, _, filenames) in os.walk(folder):
			if platform == 'linux' or platform == 'linux2' or platform == 'darwin':
				# linux and OSX
				split_path = dirpath.split('/')
			else:
				# windows
				split_path = dirpath.split('\\')
			if filenames:
				for file in filenames:
					encode_path = (f'{self.feature_folder}/{split_path[-2]}', f'{file[:14]}.pic')
					coord_path = (f'{self.config["facial_data"]}', f'{file[:14]}.pic')
					if file.endswith('.mp4') and not os.path.exists(f'{encode_path[0]}/{encode_path[1]}'):
						print(f'Extracting features from {file}')
						faces, coords = self.video_faces(f'{dirpath}/{file}', f'{coord_path[0]}/{coord_path[1]}', face_detector)
						encoding = self.vggface_encoding(faces, encoder)
						save_to_file(coord_path[0], coord_path[1], coords)
						save_to_file(encode_path[0], encode_path[1], encoding.reshape(encoding.shape[0], -1))

	def video_faces(self, video_path, coord_path, face_detector):
		"""
			 Extracts faces from a video returning array of rgb images of faces

			 Parameters
			 ----------
			 video_path : str
				 A path to the video file
				 
			 coord_path : str
			     Folder location and file name of the file with face coordinates
			     
			 face_detector : object
			     MTCNN network for face detection
			     
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

		i = -1
		while cap.isOpened():
			ret, frame = cap.read()
			if ret:
				i += 1
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				if not coords_present:
					all_coords[i] = self.get_face_coords(frame, face_detector)
				c = all_coords[i]
				if (c == -1).all():
					continue
				else:
					face = frame[c[0]:c[1], c[2]:c[3]]
					faces[i] = cv2.resize(face, (self.input_size[0], self.input_size[1])).astype('float64')
			else:
				break
		cap.release()

		return faces, all_coords
	
	@staticmethod
	def get_face_coords(frame, face_detector):
		"""
			Detects and extracts a face from the image

			Parameters
			----------
			frame : ndarray (? * ? * 3)
				An RGB array of image
				
			face_detector : MTCNN network for face detection
				
			Returns
			-------
			face : 4-tuple
				A face bounding box extracted from the original image and fitted for input size
		"""

		try:
			results = face_detector.detect_faces(frame)  # Detects faces in the image

			x1, y1, width, height = results[0]['box']  # Bounding box of first face
			x1, y1 = abs(x1), abs(y1)  # bug fix...
			x2, y2 = x1 + width, y1 + height
		except:
			x1, x2, y1, y2 = -1, -1, -1, -1
		return y1, y2, x1, x2
	
	@staticmethod
	def vggface_encoding(faces, encoder):
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
		yhat = encoder.predict(inputs)
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

		frames, components = video.shape  # (N, C) in the paper
		pattern_len = 5  # (M) in the paper
		# TODO: Investigate this value... Suggested to use if features are [0, 1]
		thresh = 1 / 255  # (T) in the paper

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

		return fdhh

	@staticmethod
	def video_min_max(folder, scaler=None):

		files = os.listdir(folder)
		if not scaler:
			scaler = MinMaxScaler()

		for file in files:
			video_data = load_from_file(f'{folder}/{file}')
			scaler.partial_fit(video_data)

		return scaler
