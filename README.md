# Darwin-Project
An advance algorithm for depression and anxiety detection from extracted facial and audio features. The implementation 
is based of several AVEC challenge participants with a number of personal improvements and additions.

## Setting up the Environment

To create the environment run:
```
conda env create -f darwin_env.yaml
```
Then activate it with
```
conda activate darwin
```
If you install any new libraries, make sure you do it with conda 
unless there is no other choice. Then, update the environment with
```
conda env export --from-history > darwin_env.yaml
```

## Running Experiments
To run you own experiment you have to first edit the `config.ini` file 
paying careful attention to naming conventions and ensuring you are not
duplicating already existing functionality with a different name.

1. Set a name for experiment
2. For a single model:
    2. Use appropriate model name (Should be same as function name in `model.py`)
    3. Set model weights to 1
    4. Use appropriate name for features
    5. var_ratio determines how many components to keep after PCA
3. For multiple models:
    1. Separate the models and there attributes by a `+` sign
    2. Make sure model weights add to **1**
4. If model has not been added yet, create a new function in `model.py` and add a 
call to it in the `switcher` statement. Make sure to use `try` `except` clause inside
you function to avoid errors.
5. If features don't exists add them to relevant feature script and then load them in 
trough `data.py` functionality. Make sure the final data is formatted correctly.
6. Run your experiment and look up the results by running `mlflow ui` in same directory
as **mlruns** folder

## Optimizing your model
You can optimize your code by running `pipelie.optimize` and selecting parameter appropriately. Use a **tuple** 
to determine **min** and **max** for each of the desired parameters. For example `(100, 1000)` will search
for the best parameter in range between **100** and **1000**. The optimization uses a Bayesian Optimization technique 
and more information can be found [here](https://github.com/fmfn/BayesianOptimization).