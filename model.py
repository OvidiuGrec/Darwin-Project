

class DepressionModel:
	
	def __init__(self, config):
		self.model_type = config['model_type']
		self.model_name = config['model_name']
	
	def get_models(self):
		
		model_switcher = {
			'PLS': PLS(),
			'LR': LR(),
		}
	
	def PLS(self):
	
	