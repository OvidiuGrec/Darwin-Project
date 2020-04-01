import pandas as pd
import os

from configparser import ConfigParser

from data import Data
from model import DepressionModel

class Pipeline:
    
    def __init__(self):
        self.config = self.load_config()
        self.data = Data(self.config)
        self.model = DepressionModel(self.config)
        
    def load_parameters(self):
        """
            Define your own parameters for the model here
    
            Returns
            -------
            pars: dict
                A multilevel dictionary of model parameters for each specified model
        """
        pars = {
            'LR': {'normalize': True},
            'PLS': {''}
        }
    
    def train_model(self):
        
        X_train, y_train, X_dev, y_dev = self.data.load_data()
        
        
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
    
        config['model_type'] = parser.get("parameters", "model_type")
        config['model_name'] = parser.get("parameters", "model_name")
        config['video_features'] = parser.get("parameters", "video_features")
        config['audio_features'] = parser.get("parameters", "audio_features")
        
        config['raw_video_folder'] = parser.get("folders", "raw_video_folder")
        config['video_folder'] = parser.get("folders", "video_folder")
        config['audio_folder'] = parser.get("folders", "audio_folder")
        config['labels_folder'] = parser.get("folders", "labels_folder")
    
        return config


if __name__ == '__main__':
    pipe = Pipeline()
    self = pipe.data
    self = self.video


