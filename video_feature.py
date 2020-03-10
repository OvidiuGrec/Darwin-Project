from os import walk, environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import cv2

from sklearn.preprocessing import MinMaxScaler

from mtcnn import MTCNN
import tensorflow as tf
from keras import Model
import keras.models as models
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input


VGGFACE_FACE_SIZE = (224, 224, 3)


def all_video_features():

	video_folder = 'data/video'
	
	face_detector = MTCNN()
	model = get_vggface()
	
	dataset = {}
	for (dirpath, _, filenames) in walk(video_folder):
		split_path = dirpath.split('\\')
		if filenames:
			folder_data = pd.DataFrame(columns=[file[:-10] for file in filenames])
			for file in filenames:
				print(f'Extracting features from {file}')
				folder_data[file] = encode_video(f'{dirpath}/{file}', face_detector, model)
			pd.to_pickle(folder_data, f'{video_folder}/features/{split_path[-2].lower()}_{split_path[-1].lower()}.pkl')
			
			
def encode_video(filename, face_detector, model):
	"""
		Encodes frames from video file into a lower dimensional representation

		Parameters
		----------
		filename : string
			Name of the file to be encoded

		Returns
		-------
		encoded_faces : ndarray (? * 128)
			A matrix representing 128 features for each frame in the video
	"""
	
	cap = cv2.VideoCapture(filename)
	
	FPS = int(cap.get(cv2.CAP_PROP_FPS))
	n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	
	# Check if camera opened successfully
	if not cap.isOpened():
		print("Error opening video stream or file")
	
	all_images = np.empty(shape=(n_frames, 224, 224, 3))
	i = 0
	while cap.isOpened():
		ret, frame = cap.read()
		if ret:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			face = extract_face(frame, face_detector)
			if face is not None:
				all_images[i] = face
			else:
				all_images[i] = all_images[i-1]
			i += 1
		else:
			break
	cap.release()
	
	encoded_faces = vggface_encoding(all_images, model)
	scaled = MinMaxScaler().fit_transform(encoded_faces)  # TODO: MinMax should be trained on all videos...
	
	fdhh = FDHH(scaled)/n_frames  # Added normalisation based on number of frames...? Worth testing
	
	return fdhh.flatten()


def extract_face(frame, face_detector, required_size=(224, 224)):
	"""
		Detects and extracts a face from the image
	
		Parameters
		----------
		frame : ndarray (? * ? * 3)
			An RGB array image
		
		face_detector :
			An MTCNN object for face detection
	
		Returns
		-------
		face : ndarray (required_size * 3)
			A face extracted from the original image and fitted into 160 by 160 pixels
	"""
	
	try:
		results = face_detector.detect_faces(frame)  # Detects faces in the image
	
		x1, y1, width, height = results[0]['box']  # Bounding box of first face
		x1, y1 = abs(x1), abs(y1)  # bug fix...
		x2, y2 = x1+width, y1+height
	
		face = frame[y1:y2, x1:x2]
		face = cv2.resize(face, required_size).astype('float64')  # Resize for FaceNet
	except:
		face = None
	return face


def get_vggface():
	
	# this returns convolution features
	# vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3), pooling='avg')
	
	# this returns layer-specific features:
	wanted_layer = "fc6"  # example layer name, see https://github.com/rcmalli/keras-vggface/blob/master/keras_vggface/models.py
	vgg_model = VGGFace(model='vgg16', weights='vggface', input_shape=(224, 224, 3), include_top=True)
	out = vgg_model.get_layer(wanted_layer).output
	# r = layers.Flatten()(out)
	vgg_model_custom_layer = Model(inputs=vgg_model.input, outputs=out)
	
	return vgg_model_custom_layer


def vggface_encoding(faces, vgg_model):
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
	
	inputs = preprocess_input(faces, version=1)  # TODO: play around with this...
	yhat = vgg_model.predict(inputs)
	return yhat


def FDHH(video_segment):

	frames, components = video_segment.shape  # (N, C) in the paper
	pattern_len = 5  # (M) in the paper
	# TODO: Investigate this value... Suggested to use if features are [0, 1]
	thresh = 1/40  # (T) in the paper

	dynamics = np.abs(video_segment[1:] - video_segment[:-1])
	binary_d = np.where(dynamics > thresh, 1, 0).T  # (D(c,n)) in the paper

	fdhh = np.zeros((components, pattern_len))

	for c in range(components):
		count = 0
		for n in range(frames - 2):
			if binary_d[c, n+1]:
				count += 1
			elif 0 < count <= pattern_len:
				fdhh[c, count-1] += 1
				count = 0
			elif count > pattern_len:
				fdhh[c, pattern_len-1] += 1
				count = 0
	
	return fdhh


if __name__ == '__main__':
	all_video_features()