import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
import tensorflow as tf
import numpy as np
import pandas as pd
import mlflow
from configparser import ConfigParser
from argparse import ArgumentParser
from functools import partial

from data import Data
from model import DepressionModel
from validation import avec_metrics, run_validation
from bayes_opt import BayesianOptimization


class Pipeline:

    def __init__(self):
        
        self.int_pars = ['l1', 'l2', 'batch_size', 'epochs', 'pattern_len', 'seq_length']
        self.no_log = ['experiment', 'seed', 'folders', 'config_location', 'feature_type']
        
        self.options = self.parse_options()
        self.config = self.load_config()
        self.pars = self.load_pars()
        self.data = Data(self.config, self.options, self.pars)
        self.fusion = self.config['combined']['fusion']
        self.feature_type = self.config['general']['feature_type']
        
        # Add configs that should not be logged
        if self.fusion == 'late':
            self.no_log += ['combined']
            
        if self.feature_type == 'video':
            self.no_log += ['audio', 'audio_features', 'combined']
        elif self.feature_type == 'audio':
            self.no_log += ['video', 'video_features', 'combined']
        elif self.feature_type == 'combined':
            self.no_log += ['audio', 'video']

    def run_experiment(self, **kwargs):
        
        if self.options.opt:
            self.optimize()
        
        seed = self.config['general']['seed']
        np.random.seed(seed)
        tf.set_random_seed(seed)

        if kwargs:
            self.adjust_pars(kwargs)
            self.data = Data(self.config, self.options, self.pars)

        if self.options.mlflow:
            mlflow.set_experiment(self.config['general']['experiment'])
            mlflow.start_run()
            self.log_pars()
            mlflow.set_tags({'seed': seed, 'mode': self.options.mode})

        if self.feature_type == 'combined' and self.fusion == 'late':
            y_true, y_pred, mae = self.train_bimodal()
        else:
            _, y_pred, _, y_true, mae = self.train_model(self.feature_type)
        
        if self.options.verbose:
            run_validation(y_true, y_pred)
            
        if self.options.mlflow:
            mlflow.end_run()

        return -mae

    def train_bimodal(self):
        v_pred_train, v_pred_test, y_train, y_test, _ = self.train_model('video')
        a_pred_train, a_pred_test, y_train, y_test, _ = self.train_model('audio')

        pred_weights = [float(v) for v in self.config['combined']['prediction_weights'].split('+')]
        pred_train = pred_weights[0] * v_pred_train + pred_weights[1] * a_pred_train
        pred_test = pred_weights[0] * v_pred_test + pred_weights[1] * a_pred_test

        train_mae, train_rmse = avec_metrics(y_train, pred_train)
        test_mae, test_rmse = avec_metrics(y_test, pred_test)

        self.log_metrics('combined', train_mae, train_rmse, test_mae, test_rmse)
        
        return y_test, pred_test, test_mae

    def train_model(self, feature_type):
        X_train, y_train, X_test, y_test = self.data.load_data(feature_type)
        input_shape = X_train.shape
        
        # self.pars['VanilaLSTM']['train']['validation_data'] = (X_test, y_test)
        model = DepressionModel(feature_type, self.config[feature_type], input_shape, pars=self.pars)
        
        model.train(X_train, y_train)
        pred_train = pd.DataFrame(data=model.predict(X_train), index=y_train.index)
        pred_test = pd.DataFrame(data=model.predict(X_test), index=y_test.index)
        
        train_mae, train_rmse = avec_metrics(y_train, pred_train)
        test_mae, test_rmse = avec_metrics(y_test, pred_test)

        self.log_metrics(feature_type, train_mae, train_rmse, test_mae, test_rmse)

        return pred_train, pred_test, y_train, y_test, test_mae

    def optimize(self):
        self.options.opt = False
        pbounds = {}
        for k0, v0 in self.pars.items():
            for k1, v1 in v0.items():
                try:
                    for k2, v2 in v1.items():
                        if isinstance(v2, (list, tuple)):
                            pbounds[k2] = v2
                except AttributeError:
                    if isinstance(v1, (list, tuple)):
                        pbounds[k1] = v1
    
        opt_part = partial(self.run_experiment, use_mlflow=False)
        optimizer = BayesianOptimization(f=opt_part, pbounds=pbounds, verbose=2, random_state=1)
        optimizer.maximize(init_points=10, n_iter=90)
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
                        
    def log_metrics(self, feature_type, train_mae, train_rmse, test_mae, test_rmse):
        if self.options.mlflow:
            log_type = f'{feature_type}_' if feature_type else ''
            mlflow.log_metrics({f'{log_type}train_mae': train_mae, f'{log_type}train_rmse': train_rmse,
                                f'{log_type}test_mae': test_mae, f'{log_type}test_rmse': test_rmse})
    
    def adjust_pars(self, new_pars):
    
        for key, value in new_pars.items():
            for k0, v0 in self.pars.items():
                for k1, v1 in v0.items():
                    try:
                        for k2, v2 in v1.items():
                            if k2 == key:
                                if k2 in self.int_pars:
                                    value = int(value)
                                v1[k2] = value
                    except AttributeError:
                        if k1 == key:
                            if k1 in self.int_pars:
                                value = int(value)
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
                            default=False,
                            help="Specify where as to save the run to mlflow experiments")
        parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", required=False,
                            help="Specify if you want to print out all of the outputs and graphs")
        parser.add_argument("--save-f", dest="save_features", action="store_true", required=False,
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
        general['feature_type'] = parser.get('general', 'feature_type')
        general['audio_features'] = parser.get("general", "audio_features")
        general['video_features'] = parser.get("general", "video_features")

        combined = config['combined']
        combined['combined_model'] = parser.get("combined", "combined_model")
        combined['combined_model_weights'] = parser.get("combined", "combined_model_weights")
        combined['fusion'] = parser.get("combined", "fusion")
        combined['prediction_weights'] = parser.get("combined", "prediction_weights")
        combined['combined_scaler'] = parser.get("combined", "combined_scaler")
        combined['combined_scale_axis'] = parser.get("combined", 'combined_scale_axis')
        
        audio = config['audio']
        audio['audio_model'] = parser.get("audio", "audio_model")
        audio['audio_model_weights'] = parser.get("audio", "audio_model_weights")
        audio['audio_scaler'] = parser.get("audio", "audio_scaler")
        audio['audio_scale_axis'] = parser.get("audio", 'audio_scale_axis')

        video = config['video']
        video['video_model'] = parser.get("video", "video_model")
        video['video_model_weights'] = parser.get("video", "video_model_weights")
        video['video_scaler'] = parser.get("video", "video_scaler")
        video['video_scale_axis'] = parser.get("video", 'video_scale_axis')
        
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
        return pars


if __name__ == '__main__':
    pipe = Pipeline()
    self = pipe
    # pipe.run_experiment()
