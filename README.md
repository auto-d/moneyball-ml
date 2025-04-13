# moneyball-ml
MLB statcast data vs conventional baseball metrics in win prediction

# TODO : 

[ ] implemeent the MBDataset class so it returns the data in the right format 
[ ] pass the dataset output in the `evaluate` method when we're operating on an 'nn' pipeline
[ ] get some damned results for the NN 
[ ] generate visualizations of cluster, algorithm progress, etc... to add to presentation!! 
[ ] complete dataflow /pipeline diagram and incorporate!

From Week 12 lecture notes: 
```
[ ] 10-minute max video presentation that addresses
    [ ] Data and data prep pipeline
    [ ] Feature engineering and selection
    [ ] Choice of evaluation metric / justification
    [ ] Evaluation strategy
    [ ] Model optimization
    [ ] Must describe both DL and non-DL models
    [ ] Comment on model performance
        [ ] Expected / unexpected
        [ ] What could be done to improve it

[ ] Project code repo
```

Notes: 
- how does a player's performance in one season predict their performance the next season?
    - offer up a simplified version: 2023 / 2024
        - try to predict WAR for 2024 
            - three models: 
                - statistics from 2023 (to predict 2024 WAR)
                - model statistics 
                - statcast + statistics 
                - injuries accounted for -- plate appearances, etc.... or pitchers 
                    - only include the union of the players 
                - clean regression problem
- less about results and more about approach


**Goal**: Model the role of Major League Baseball (MLB) Statcast data in predicting player value. 

**Evaluation Criteria**: 

**Metric Selection**: 

The objective is to predict the WAR value for players within some reasonable margin. A few tenths off one way or another is probably a great prediction, but being off by an integer or two is problematic. We settle on **mean squared error** as a metric to disproportionately penalize models that produce a higher variance in their predictions. 

## Evaluation strategy 

Linear Regression, with L1 and L2 penalties 
Support Vector Regressor
Random Forest Regressor 
Hyperparameter grid search across wide range of SKLearn regressors and neural networks. 

Cross validation w



## Approach

The approach to the prediction task is outlined below.

1. 
2

retrospective: 
bin players by position, then create different models, or at least parameters to predict? 

## Exploratory Data Analysis



## Data Preparation 

The 

- outliers, missing data, corrupted data, hetergeneity of reporting, data types, 
- importance of homogenization, intersecting and joining to ensure 
- sparsity due to one-hot encodings.. 
- note challenges with feature count
    - sparsity of one-hot encoding for categoricals
    - presence of nans

- discard conventional metrics that betray outcomes such as win rate, clutch performances, etc...


## Feature Engineering 


## Data Pipeline


## Algorithm and Hyperparameter Search


