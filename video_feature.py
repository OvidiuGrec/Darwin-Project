import numpy as np
import cv2

import tensorflow.keras.models as models
from mtcnn import MTCNN


def encode_video(filename):
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
	
	face_detector = MTCNN()
	
	cap = cv2.VideoCapture(filename)
	
	# Check if camera opened successfully
	if not cap.isOpened():
		print("Error opening video stream or file")
	
	all_faces = []
	while cap.isOpened():
		ret, frame = cap.read()
		if ret:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			face = extract_face(frame, face_detector)
			all_faces.append(face)
		else:
			break
	cap.release()
	
	all_faces = np.array(all_faces)
	encoded_faces = facenet_encoding(all_faces)
	
	return encoded_faces


def extract_face(frame, face_detector):
	"""
		Detects and extracts a face from the image
	
		Parameters
		----------
		frame : ndarray (? * ? * 3)
			An RGB array image
		
		face_detector : object
			An MTCNN object for face detection
	
		Returns
		-------
		face : ndarray (160 * 160 * 3)
			A face extracted from the original image and fitted into 160 by 160 pixels
	"""
	
	results = face_detector.detect_faces(frame)  # Detects faces in the image

	x1, y1, width, height = results[0]['box']  # Bounding box of first face
	x2, y2 = x1+width, y1+height

	face = frame[y1:y2, x1:x2]
	face = cv2.resize(face, (160, 160))  # Resize for FaceNet
	
	return face.astype('float64')


def facenet_encoding(faces):
	"""
		Encodes an image of a face into a lower dimensional representation

		Parameters
		----------
		faces : ndarray (n * 160 * 160 * 3)
			An array of face images to be encoded

		Returns
		-------
		encoding : ndarray (n * 128)
			Feature-vector of a face representations
	"""
	
	scaled_faces = (faces - faces.mean()) / faces.std()  # Standardise across channels
	
	facenet = models.load_model('models/facenet_keras.h5', compile=False)
	encoding = facenet.predict(scaled_faces)
	
	return encoding


def FDHH(video_segment):

	frames, components = video_segment.shape  # (N, C) in the paper
	pattern_len = 5  # (M) in the paper
	# TODO: Investigate this value... Suggested to use if features are [0, 1]
	thresh = 1/255  # (T) in the paper

	dynamics = np.abs(video_segment[1:] - video_segment[:-1])
	binary_d = np.where(dynamics > thresh, 1, 0).T  # (D(c,n)) in the paper

	fdhh = np.zeros((pattern_len, components))

	for c in range(components):
		count = 0
		for n in range(frames - 2):
			if binary_d[c, n+1]:
				count += 1
			elif 0 < count <= pattern_len:
				fdhh[count-1, c] += 1
				count = 0
			elif count > pattern_len:
				fdhh[pattern_len-1, c] += 1




	












