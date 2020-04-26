import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
import tensorflow as tf
import numpy as np
import mlflow
from configparser import ConfigParser
from argparse import ArgumentParser
from functools import partial

from data import Data
from model import DepressionModel
from bayes_opt import BayesianOptimization


class Pipeline:

    def __init__(self):
        
        self.int_pars = ['l1', 'l2', 'batch_size', 'epochs']
        self.no_log = ['experiment', 'seed', 'folders', 'config_location']
        
        self.options = self.parse_options()
        self.config = self.load_config()
        self.pars = self.load_pars()
        self.data = Data(self.config, self.options, self.pars)
        self.fusion = self.config['general']['fusion']
        if self.fusion == 'late':
            self.no_log.append('combined')

    def run_experiment(self, **kwargs):
        
        if self.options.opt:
            self.optimize()
        
        seed = self.config['general']['seed']
        np.random.seed(seed)
        tf.set_random_seed(seed)
        
        if kwargs:
            self.adjust_pars(kwargs)

        X_train, y_train, X_test, y_test = self.data.load_data()
        input_shape = X_train.shape
        
        if self.options.mlflow:
            mlflow.set_experiment(self.config['general']['experiment'])
            mlflow.start_run()
            self.log_pars()
            mlflow.set_tags({'seed': seed})
                           
        model = DepressionModel(self.config['combined'], input_shape, pars=self.pars)
        model.train(X_train, y_train)
        # TODO: log metrics through out the run
        
        train_mae, train_rmse = model.validate_model(X_train, y_train)
        dev_mae, dev_rmse = model.validate_model(X_test, y_test)

        if self.options.mlflow:
            mlflow.log_metrics({'train_mae': train_mae, 'train_rmse': train_rmse,
                                'dev_mae': dev_mae, 'dev_rmse': dev_rmse})
            mlflow.end_run()
        
        return -dev_mae

    def optimize(self):
        pbounds = {}
        for k0, v0 in self.pars.items():
            for k1, v1 in v0.items():
                for k2, v2 in v1.items():
                    if isinstance(v2, (list, tuple)):
                        pbounds[k2] = v2
    
        opt_part = partial(self.run_experiment, use_mlflow=False)
        optimizer = BayesianOptimization(f=opt_part, pbounds=pbounds, verbose=2, random_state=1)
        optimizer.maximize(init_points=10, n_iter=100)
        for i, res in enumerate(optimizer.res):
            print("Iteration {}: \n\t{}".format(i, res))
        print(optimizer.max)
        
    def log_pars(self):
        # Log parameters from pars file:
        for k0, v0 in self.pars.items():
            for k1, v1 in v0.items():
                try:
                    for k2, v2 in v1.items():
                        mlflow.log_param(f'{k0}_{k1}_{k2}', v2)
                except AttributeError:
                    mlflow.log_param(f'{k0}_{k1}', v1)
        
        # Log parameters from config:
        for k0, v0 in self.config.items():
            if k0 not in self.no_log:
                for k1, v1 in v0.items():
                    if k1 not in self.no_log:
                        mlflow.log_param(k1, v1)
    
    def adjust_pars(self, new_pars):
    
        for key, value in new_pars.items():
            for k0, v0 in self.pars.items():
                for k1, v1 in v0.items():
                    try:
                        for k2, v2 in v1.items():
                            if k2 == key:
                                v1[k2] = value
                    except AttributeError:
                        if k1 == key:
                            v0[k1] = value
    
    @staticmethod
    def parse_options():
        
        parser = ArgumentParser(description="This is a command line interface (CLI) for the Darwin Project",
                                epilog="Vlad Bondarenko, Andrei Clopotel, Ovidiu Grec, Tymon Soleci 2020-04-26")
        parser.add_argument("-c", "--config", dest="config", action="store", type=str, required=False,
                            metavar="<path-to-file", default="Config/default_config.ini",
                            help="Specify the location of config file you want to use (Make sure it has "
                                 "identical fields to the default_config.ini.")
        parser.add_argument("-p", "--pars", dest="pars", action="store", type=str, required=False,
                            metavar="<path-to-file", default="Config/default_pars.json",
                            help="Specify the location of parameters to use with the model.")
        parser.add_argument("-m", "--mode", dest="mode", action="store", type=str, required=False,
                            metavar="<mode-to-run-in>", default="dev",
                            help="Specify a mode to run in from (dev or test). (default: dev)")
        parser.add_argument("-o", "--opt", dest="opt", action="store_true", required=False,
                            help="Specify if you want to run hyperparameter optimization")
        parser.add_argument("-s", "--save", dest="mlflow", action="store_true", required=False,
                            help="Specify where as to save the run to mlflow experiments")
        parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", required=False,
                            help="Specify if you want to print out all of the outputs and graphs")
        parser.add_argument("--save-fdhh", dest="save_fdhh", action="store_true", required=False,
                            help="Specify where as to store the FDHH features or not")
        
        options = parser.parse_args()
        return options
    
    def load_config(self):
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

        file_location = self.options.config

        parser = ConfigParser()
        parser.read(file_location)

        config = dict()
        config['general'] = dict()
        config['audio'] = dict()
        config['video'] = dict()
        config['combined'] = dict()
        config['folders'] = dict()
        
        general = config['general']
        general['config_location'] = file_location
        general['experiment'] = parser.get("general", "experiment")
        general['seed'] = parser.getint("general", "seed")
        general['fusion'] = parser.get("general", "fusion")
        
        audio = config['audio']
        audio['audio_features'] = parser.get("audio", "audio_features")
        
        video = config['video']
        video['video_features'] = parser.get("video", "video_features")
        
        combined = config['combined']
        combined['model_name'] = parser.get("combined", "model_name")
        combined['model_weights'] = parser.get("combined", "model_weights")
        
        folders = config['folders']
        folders['raw_video_folder'] = parser.get("folders", "raw_video_folder")
        folders['facial_data'] = parser.get("folders", "facial_data")
        folders['video_folder'] = parser.get("folders", "video_folder")
        folders['raw_audio_folder'] = parser.get("folders", "raw_audio_folder")
        folders['seg_audio_folder'] = parser.get("folders", "seg_audio_folder")
        folders['audio_folder'] = parser.get("folders", "audio_folder")
        folders['labels_folder'] = parser.get("folders", "labels_folder")
        folders['models_folder'] = parser.get("folders", "models_folder")

        return config
    
    def load_pars(self):
        with open(self.options.pars, 'r') as f:
            pars = json.load(f)
        
        # Convert all lists to tuples for optimization
        for k0, v0 in pars.items():
            for k1, v1 in pars[k0].items():
                try:
                    for k2, v2 in v0[k1].items():
                        if isinstance(v2, list):
                            v1[k2] = tuple(v2)
                except AttributeError:
                    if isinstance(v1, list):
                        v0[k1] = tuple(v1)
                        
        return pars


if __name__ == '__main__':
    pipe = Pipeline()
    self = pipe
    # pipe.optimize()
    pipe.run_experiment()
