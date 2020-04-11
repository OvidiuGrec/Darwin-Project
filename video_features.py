import os
import cv2
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from helper import save_to_file, load_from_file

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from mtcnn import MTCNN
from keras import Model
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input


class VideoFeatures:
	
	def __init__(self, config):
		self.face_detector = MTCNN()
		self.input_size = (224, 224, 3)
		self.config = config
		self.vgg_v, self.vgg_l, fdhh = self.config['video_features'].split('_')
		self.model = self.get_vggface()
		self.fdhh = bool(fdhh)
		self.feature_folder = f'{self.config["video_folder"]}/{self.vgg_v}_{self.vgg_l}'
		
	def get_video_data(self, data_part):
		"""
			Retruns video feature data depending on parameter provided in config file.
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
		
		folder = self.config['raw_video_folder']
		
		for (dirpath, _, filenames) in os.walk(folder):
			split_path = dirpath.split('\\')
			if filenames:
				for file in filenames:
					file_path = (f'{self.feature_folder}/{split_path[-2]}', f'{file[:14]}.pic')
					if file.endswith('.pic') and not os.path.exists(f'{file_path[0]}/{file_path[1]}'):
						print(f'Extracting features from {file}')
						faces = self.video_faces(f'{dirpath}/{file}')
						encoding = self.vggface_encoding(faces)
						save_to_file(file_path[0], file_path[1], encoding)
				
	def video_faces(self, video_file):
		"""
			Extracts faces from a video returning array of rgb images of faces

			Parameters
			----------
			video_file : str
				A path to the video file

			Returns
			-------
			faces : ndarray (? * ? * ? * 3)
				An array of faces corresponding to each frame in the video. Leaves nan values if a face is missing
		"""
		
		cap = cv2.VideoCapture(video_file)
		n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		
		# Check if camera opened successfully
		if not cap.isOpened():
			print("Error opening video stream or file")
		
		faces = np.empty(shape=(n_frames, self.input_size[0], self.input_size[1], self.input_size[2]))
		faces[:] = np.nan
		
		i = 0
		while cap.isOpened():
			ret, frame = cap.read()
			if ret:
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				face = self.extract_face(frame)
				if face is not None:
					faces[i] = face
				i += 1
			else:
				break
		cap.release()
		
		return faces
	
	def extract_face(self, frame):
		"""
			Detects and extracts a face from the image

			Parameters
			----------
			frame : ndarray (? * ? * 3)
				An RGB array of image
				
			Returns
			-------
			face : ndarray (required_size * 3)
				A face extracted from the original image and fitted for input size
		"""
		
		try:
			results = self.face_detector.detect_faces(frame)  # Detects faces in the image
			
			x1, y1, width, height = results[0]['box']  # Bounding box of first face
			x1, y1 = abs(x1), abs(y1)  # bug fix...
			x2, y2 = x1 + width, y1 + height
			
			face = frame[y1:y2, x1:x2]
			face = cv2.resize(face, (self.input_size[0], self.input_size[1])).astype('float64')  # Resize for VGGFace
		except:
			face = None
		return face
	
	def vggface_encoding(self, faces):
		"""
			Encodes an image of a face into a lower dimensional representation
			Parameters
			----------
			faces : ndarray (n * 224 * 224 * 3)
				An array of face images to be encoded
			Returns
			-------
			encoding : ndarray (n * X)
				Feature-vector of a face representations
		"""
		
		inputs = preprocess_input(faces, version=1)  # TODO: 1 for VGG and 2 for others
		yhat = self.model.predict(inputs)
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
			'35': 'fc7/relu'
		}
		
		model_select = {
			'VGG': 'vgg16',
			'RES': 'resnet50',
			'SE': 'senet50'
		}
		
		# this returns layer-specific features:
		wanted_layer = layers_select[self.vgg_l]
		vgg_model = VGGFace(model=model_select[self.vgg_v], input_shape=(224, 224, 3), include_top=True)
		
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
	
	def video_min_max(self, folder):
		
		files = os.listdir(folder)
		scaler = MinMaxScaler()
		
		for file in files:
			video_data = load_from_file(f'{folder}/{file}')
			scaler.partial_fit(video_data)
		
		return scaler
