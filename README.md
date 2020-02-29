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
