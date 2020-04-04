import mlflow
from configparser import ConfigParser

from data import Data
from model import DepressionModel

class Pipeline:
    
    def __init__(self):
        self.config = self.load_config()
        self.data = Data(self.config)
        self.pars = self.load_pars()
        self.model = DepressionModel(self.config, pars=self.pars)
    
    @staticmethod
    def load_pars():
        """
            Define your own parameters for the model here
    
            Returns
            -------
            pars: dict
                A multilevel dictionary of model parameters for each specified model
        """
        pars = {
            'LR': {'model': {'normalize': True}},
            'PLS': {'model': {'n_components': 2}}
        }
        """
        Example pars for neural network:
        pars = {
            'FNN' : {'model': {'l1': 256, 'l2': 256},
                     'train': {'epochs': 100, 'batch_size': 64}}
        }
        """
        return pars
    
    def run_experiment(self):
        
        mlflow.set_experiment(self.config['experiment'])
        with mlflow.start_run():
            
            mlflow.log_params({'pars': str(self.pars),
                               'model': self.config['model_name'],
                               'weight': self.config['model_weights'],
                               'video': self.config['video_features'],
                               'audio': self.config['audio_features'],
                               'n_in': str(self.config['n_in'])})
            
            model = self.model
            
            X_train, y_train, X_dev, y_dev = self.data.load_data()
            model.train(X_train, y_train)
            
            # TODO: log metrics through out the run
            train_mae, train_rmse = model.validate_model(X_train, y_train)
            dev_mae, dev_rmse = model.validate_model(X_dev, y_dev)
            
            mlflow.log_metrics({'train_mae': train_mae, 'train_rmse': train_rmse,
                                'dev_mae': dev_mae, 'dev_rmse': dev_rmse})
        
    @staticmethod
    def load_config(file_location=None):
        """
        Load the config from file.

        Parameters
        ----------
        `file_location` : string
            Optionally provide the location of the config file as full absolute path. If not
            provided, config is assumed to be in 'Config/config.ini'.
        Returns
        -------
        dict
            A dictionary of config parameters whose keys match the names used in the config file.
        """
        if not file_location:
            file_location = 'Config/config.ini'
    
        parser = ConfigParser()
        parser.read(file_location)
    
        config = dict()
        config['config_location'] = file_location
        
        config['experiment'] = parser.get("parameters", "experiment")
        config['model_name'] = parser.get("parameters", "model_name")
        config['model_weights'] = parser.get("parameters", "model_weights")
        config['video_features'] = parser.get("parameters", "video_features")
        config['audio_features'] = parser.get("parameters", "audio_features")
        config['n_in'] = parser.getint("parameters", "n_in")
        
        config['raw_video_folder'] = parser.get("folders", "raw_video_folder")
        config['facial_data'] = parser.get("folders", "facial_data")
        config['video_folder'] = parser.get("folders", "video_folder")
        config['audio_folder'] = parser.get("folders", "audio_folder")
        config['labels_folder'] = parser.get("folders", "labels_folder")
    
        return config


if __name__ == '__main__':
    self = Pipeline()


