from keras import Model
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

class MyVGGFace:
	
	def __init__(self, vgg_l, vgg_v):
		
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
		
		self.layer = layers_select[vgg_l]
		self.model = model_select[vgg_v]
		self.version = 1 if vgg_v == 'VGG' else 2
		self.include_top = True if vgg_v == 'VGG' else False
		self.pooling = 'avg' if vgg_v != 'VGG' else None
		
		self.encoder = self.get_vggface()
		
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
		
		inputs = preprocess_input(faces, version=self.version)
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
		
		# this returns layer-specific features:
		vgg_model = VGGFace(model=self.model, input_shape=(224, 224, 3), include_top=self.include_top,
		                    pooling=self.pooling)
		
		out = vgg_model.get_layer(self.layer).output
		vgg_model_custom_layer = Model(inputs=vgg_model.input, outputs=out)
		
		return vgg_model_custom_layer