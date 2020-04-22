import os
import numpy as np
import mlflow
from configparser import ConfigParser
from argparse import ArgumentParser
from functools import partial

from data import Data
from model import DepressionModel
from bayes_opt import BayesianOptimization

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Pipeline:

    def __init__(self):
        self.config = self.load_config()
        self.data = Data(self.config)
        self.pars = self.load_pars()
    
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
            'LR': {'model': {'normalize': True}, 'train': {}},
            'PLS': {'model': {'n_components': 5}, 'train': {}}
        }
        
        """
        pars = {
            'FNN': {'model': {'l1': 1000, 'd1': 0.2, 'lr': 0.007},
                    'train': {'epochs': 1000, 'batch_size': 1, 'validation_split': 0.2, 'verbose': 1}}
        }
        """
        # Example pars for neural network optimisation:
        """
        pars = {
            'FNN': {'model': {'l1': (10, 2000), 'd1': (0, 0.5), 'lr': (0.0001, 0.01)},
                    'train': {'epochs': (10, 1000), 'batch_size': (1, 10), 'verbose': 0}}
        }
        #"""
        return pars

    def run_experiment(self, use_mlflow=True, **kwargs):
        
        np.random.seed(666)
        
        if kwargs:
            self.adjust_pars(kwargs)

        X_train, y_train, X_dev, y_dev = self.data.load_data()
        input_shape = X_train.shape
        
        if use_mlflow:
            mlflow.set_experiment(self.config['experiment'])
            mlflow.start_run()
            mlflow.log_params({'pars': str(self.pars),
                               'model': self.config['model_name'],
                               'weight': self.config['model_weights'],
                               'video': self.config['video_features'],
                               'audio': self.config['audio_features'],
                               'var_ratio': str(self.config['var_ratio']),
                               'n_features': str(input_shape[1])})
                           
        model = DepressionModel(self.config, input_shape, pars=self.pars)
        model.train(X_train, y_train)
        # TODO: log metrics through out the run
        
        train_mae, train_rmse = model.validate_model(X_train, y_train)
        dev_mae, dev_rmse = model.validate_model(X_dev, y_dev)

        if use_mlflow:
            mlflow.log_metrics({'train_mae': train_mae, 'train_rmse': train_rmse,
                                'dev_mae': dev_mae, 'dev_rmse': dev_rmse})
            mlflow.end_run()
        
        return -dev_mae

    def optimize(self):
        pbounds = {}
        for k0, v0 in self.pars.items():
            for k1, v1 in v0.items():
                for k2, v2 in v1.items():
                    if type(v2) == list or type(v2) == tuple:
                        pbounds[k2] = v2
    
        opt_part = partial(self.run_experiment, use_mlflow=False)
        optimizer = BayesianOptimization(f=opt_part, pbounds=pbounds, verbose=2, random_state=1)
        optimizer.maximize(init_points=10, n_iter=100)
        for i, res in enumerate(optimizer.res):
            print("Iteration {}: \n\t{}".format(i, res))
        print(optimizer.max)
    
    def adjust_pars(self, new_pars):
    
        int_pars = ['l1', 'l2', 'batch_size', 'epochs']
    
        model_name = self.config['model_name']
        model_pars = self.pars[model_name]['model']
        train_pars = self.pars[model_name]['train']
    
        for key, value in new_pars.items():
            if key in int_pars:
                value = int(value)
            if key in model_pars:
                model_pars[key] = value
            elif key in train_pars:
                train_pars[key] = value
        
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
        config['var_ratio'] = parser.getfloat("parameters", "var_ratio")
        
        config['raw_video_folder'] = parser.get("folders", "raw_video_folder")
        config['raw_audio_folder'] = parser.get("folders", "raw_audio_folder")
        config['seg_audio_folder'] = parser.get("folders", "seg_audio_folder")
        config['facial_data'] = parser.get("folders", "facial_data")
        config['video_folder'] = parser.get("folders", "video_folder")
        config['audio_folder'] = parser.get("folders", "audio_folder")
        config['labels_folder'] = parser.get("folders", "labels_folder")
        config['models_folder'] = parser.get("folders", "models_folder")

        return config


if __name__ == '__main__':
    pipe = Pipeline()
    # pipe.optimize()
    # pipe.run_experiment()
