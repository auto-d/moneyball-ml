import re 
import argparse
from datetime import datetime
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn import metrics
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.pipeline import Pipeline
import torch.optim as optim
from torch.nn import MSELoss 
from torch.optim import SGD
import torch.cuda
from skorch import NeuralNetRegressor
from skorch.helper import SkorchDoctor
from data import load_statcast, load_standard, find_na
from nn import WARNet, MBDataset

def gpu_survey(): 
    """
    Inventory cuda devices, hat tip to https://stackoverflow.com/questions/48152674/how-do-i-check-if-pytorch-is-using-the-gpu
    """
    gpus = torch.cuda.is_available()
    if gpus: 
        print("GPU acceleration appears to be available!")
        for device in range(1, torch.cuda.device_count()): 
            print(f"Found {torch.cuda.get_device_name(device)}")
    else: 
        print("GPU resources not found!")

    return gpus

def min_max_scale(df, column, range=(0,1)): 
    """
    Scale a column in a DF to the provided range     

    NOTE: written for the Kaggle comp
    """
    scaler = MinMaxScaler(feature_range=range)
    scaled = scaler.fit_transform(df[[column]]) 
    return scaled.transpose()[0]

def scale(df, range=(0,1), omit=[]):
    """
    Scale a column into a given range, omitting one or more columns if provided 
    
    NOTE: utilty function written for the Kaggle comp
    """
    for column in df.columns: 
        if column not in omit: 
            df[column] = min_max_scale(df, column, (0,1))

    return df

# TODO: these have all been rewritten, test

def pca(df, components=2): 
    """
    Compress the provided dataframe into n dimensions, returning the new dims 
    as columns in a new dataframe 
    
    Note scaling is implicit
    """
    # TODO: PCA expects a mean of 0 and a variance of 1, switch this out for sklearn.StandardScaler
    scaled_df = scale(df.copy())

    pca = PCA(n_components=components) 
    pca.fit(scaled_df)
    
    dim_df = pd.DataFrame()
    for i,dimension in enumerate(pca.transform(scaled_df).T): 
        df[f'd{i}'] = dimension

    return dim_df

def kmeans(df, clusters=3):
    """
    Cluster provided DF, returning a DF with the cluster labels
    """
    km_model = KMeans(n_clusters=clusters, random_state=0, n_init="auto")
    km_model.fit(df) 

    df_cluster = pd.DataFrame()
    df_cluster['km_label'] = km_model.labels_

    return df_cluster, km_model.cluster_centers_

def dbscan(df, eps=0.5, min_samples=10): 
    """
    Compress the provided dataframe into 2 dimensions to support visualization, 
    note scaling is implicit

    """
    db = DBSCAN(eps=eps, min_samples=min_samples)
    db.fit(df)

    df_cluster = pd.DataFrame() 
    df_cluster['db_label'] = db.labels_ 

    return df_cluster

# TODO: rewrite with the new methods above
def project_results_2d(X_train, y_train, probs, threshold=0.5, clusters=8): 
    """
    Use PCA to prepare a flattened version of the training data and our performance on the TRAINING data predictions

    NOTE: utilty function written for the Kaggle comp
    """
    X_train_2d, cluster_centers = apply_pca2(X_train, clusters)
    
    # Create a subset just for training data, and then subsets for the classes
    viz_df = X_train_2d.join(y_train, how='inner', rsuffix='_y') 

    viz_df['preds'] = [ True if prob > threshold else False for prob in probs]
    viz_df['result'] = viz_df.apply(lambda x: categorize_prediction(x.label, x.preds), axis=1)

    return viz_df, cluster_centers

# TODO: rewrite to work with the new methods above
def visualize_results_2d(viz_df, cluster_centers, title, c_filter=None): 
    """
    Plot the 2d visualization, optionally with cluster centroids and/or a cluster 
    centroid filter

    NOTE: this is a utilty function written for the Kaggle comp
    """
    if c_filter is not None and c_filter != []: 
        viz_df = viz_df[viz_df['cluster'].isin(c_filter)]

    tp_sub = viz_df[viz_df['result'] == "TP"]
    fn_sub = viz_df[viz_df['result'] == "FN"]
    fp_sub = viz_df[viz_df['result'] == "FP"]
    tn_sub = viz_df[viz_df['result'] == "TN"]
    
    fig = plt.figure()     
    fig.set_size_inches(16,10) 
    
    plt.title(title)
    plt.scatter(tn_sub['d1'], tn_sub['d2'], color='gray', marker='o', label='TN')     
    plt.scatter(tp_sub['d1'], tp_sub['d2'], color='blue', marker='o', label='TP')     
    plt.scatter(fn_sub['d1'], fn_sub['d2'], color='orange', marker='.', label='FN') 
    plt.scatter(fp_sub['d1'], fp_sub['d2'], color='red', marker='.', label='FP') 

    # Note we are not plotting the DBscan clusters here, something to improve on 
    if cluster_centers is not None: 
        for cluster in range(0,len(cluster_centers)):  
            if c_filter is None or cluster in c_filter: 
                center = cluster_centers[cluster] 
                plt.scatter(center[0], center[1], color='yellow', marker='D', label='Centroids') 
                plt.annotate(cluster, (center[0], center[1]), bbox=dict(boxstyle="round", fc="0.8"))

    plt.show()

def build_train_test_set(standard=True, statcast=True):
    """
    Load, transform, and apply engineered features, returning the training set. 

    We can either assemble the sets from standard data, statcast or both (but not 
    neither :p )
    """

    if not standard and not statcast: 
        raise ValueError("One of standard or statcast must be True")
    
    X_train = pd.DataFrame()

    # Create training sets for standard and statcast data, holding WAR out as a label
    # regardless of which dataset(s) will be used
    std23 = load_standard(2023)
    std24 = load_standard(2024)

    y_train = std23['WAR']
    y_test = std24['WAR']    

    sc23 = load_statcast(2023) if statcast else None
    sc24 = load_statcast(2024) if statcast else None
    
    if standard and not statcast:
        X_train = std23.drop(['WAR'], axis=1)
        X_test = std24.drop(['WAR'], axis=1)
    elif statcast and not standard: 
        X_train = sc23
        X_test = sc24
    else: 
        # TODO: test joins 
        X_train = std23.drop(['WAR'], axis=1)
        X_train = X_train.join(sc23, how="left")
        X_test = std24.drop(['WAR'], axis=1)
        X_test = X_test.join(sc24, how="left")
    
    if len(X_train) != len(y_train) or len(X_test) != len(y_test) or (len(X_train.columns) != len(X_test.columns)):
        raise ValueError("Inconsistent data and label lengths, aborting run!")

    X_train.fillna(0., inplace=True)
    X_test.fillna(0., inplace=True)

    # Canary 
    find_na(X_train)
    find_na(X_test)

    return X_train, y_train, X_test, y_test

@ignore_warnings(category=ConvergenceWarning)
def evaluate(experiments, X_train, y_train, threshold=0.2, visualize=False): 
    """
    Search for optimal models. No internal cross validation is performed, it's expected to be
    implicit in the provided experiment pipeline execution. Ensure validation happens outside 
    of this process for models that don't incorporate validation into the associated pipeline.

    Returns the experiments (sklearn pipelines) whose error falls below the provided threshold
    """
    winners = []

    X_train_nn = scale(X_train).to_numpy().astype(np.float32)
    y_train_nn = y_train.to_numpy().astype(np.float32)
    y_train_nn = np.expand_dims(y_train_nn, axis=1)

    for i, experiment in enumerate(experiments):         

        # If the estimator in the experiment is a neural network, instantiate a torch Dataset to 
        # maximize our control over the skorch data transformation, otherwise pass DF as is. Either 
        # note we're relying on gridsearch to select the optimal model config w/ cross validation. 
        # We then do a lazy test on the TRAINING data here to log the score on the whole set. These
        # will be cross validated again before selection and moving on to face any test data. 
        preds = None
        keys = experiment.named_steps.keys()
        if 'nn' in keys or 'gridnn' in keys:             
            experiment.fit(X_train_nn, y_train_nn)
            preds = experiment.predict(X_train_nn)
        else: 
            experiment.fit(X_train, y_train)
            preds = experiment.predict(X_train)

        mse = metrics.mean_squared_error(y_train, preds)

        print(f"==== \nExperiment {i}: {mse}\n")

        if mse < threshold: 
            winners.append(experiment)
    
        if visualize: 
            # TODO: fix this  
            viz_df, centroids = project_results_2d(X_train, y_train, preds, threshold=0.5, clusters=5)
            visualize_results_2d(viz_df, centroids, title=f"Pipeline {i}, Model: {str(get_estimator_from_experiment(experiment))}", c_filter=[])

    return winners

def validate(X_train, y_train, candidates, visualize=False, splits=5): 
    """
    Validate the candidate experiments (sklearn pipelines) with cross validation
    and return a winner. 
    """
    kf = KFold(n_splits=splits, shuffle=False)

    winner_mae = None
    winner = None
    
    # skorch will eat a torch DataSet if passed directly to model.fit, but sklearn's transformations 
    # have no idea how to deal with a DataSet object. This presents a problem for transformations in the 
    # experiment paradigm used elsewhere here. We should write an sklearn transformer that can handle the 
    # DataSet, but since our only transformation is scaling, just apply it manually here (external to the 
    # sklearn experimenet pipeline) as well as in the evaluation logic above (see evaluate())
    X_train_scaled = scale(X_train)
    
    for candidate in candidates: 

        mse = None
        for train_ix, test_ix in kf.split(X_train, y_train):            
            
            preds = None 

            # NN-specific dataset transformation required before train/predict here       
            if 'nn' in candidate.named_steps.keys(): 
                
                trainset = MBDataset(X_train_scaled.iloc[train_ix], y_train.iloc[train_ix])
                testset = MBDataset(X_train_scaled.iloc[test_ix], y_train.iloc[test_ix])
                candidate.fit(trainset) 
                preds = candidate.predict(testset)
            else: 
                candidate.fit(X_train.iloc[train_ix], y_train.iloc[train_ix])
                preds = candidate.predict(X_train.iloc[test_ix])

            fold_mse = metrics.mean_squared_error(y_train.iloc[test_ix], preds)
            mse = (mse if mse is not None else 0) + fold_mse/splits

        print(f"======== \nCandidate {get_estimator_from_experiment(candidate)}: {mse}\n")
        
        if visualize: 
            # TODO: fix - this isn't classification based and the PCA support functions were refactored
            preds = candidate.predict(X_train)

            viz_df, centroids = project_results_2d(X_train, y_train, preds, threshold=0.5, clusters=5)
            visualize_results_2d(viz_df, centroids, title=f"Model: {str(get_estimator_from_experiment(candidate))} @ {mse}", c_filter=None)
        
        if (winner_mae is None) or (winner_mae > mse):
            winner_mae = mse 
            winner = candidate 
    
    return winner, winner_mae
    
def get_estimator_from_experiment(experiment):
    """
    Retrieve the model or best estimator from a fit pipeline (we use a specific tag to ID them)

    NOTE: repurposed from kaggle comp
    """
    keys = experiment.named_steps.keys()

    if 'model' in keys or 'nn' in keys:
        model = experiment.named_steps['model']
    elif 'grid' in keys or 'gridnn' in keys: 
        # Note this requires refit param to grid search to be set True
        model = experiment.named_steps['grid'].best_estimator_
    else: 
        raise ValueError("Cannot extract estimator from pipeline!")

    return model 

def generate_experiments(nn_input_size=None):
    """
    Wrap the creation of our static experiment manifest, and handle any dynamic aspects of experiment setup. Return an 
    list of experiments to run. This would be more elegant as a class but ... time. 
    
    This register will grow and shrink as experiments are run and models (w/ preferred hyperparameters) are identified. 
    
    NOTE estimators must be tagged as follows to enable model recovery and dataset transformations
     - 'model' for individual sklearn estimators 
     - 'grid' for GridSearch'd estimators
     - 'nn' for skorch/pytorch neural network estimators
     - 'gridnn' for GridSearch'd skorch nn estimators
    """

    lr_hparams = { 'alpha' : [2**x for x in range(0,10,1) ] }
    sv_hparams = { 'C' : [0.2, 0.4], 'kernel' : ['poly', 'rbf' ] }
    rf_hparams = { 'min_samples_leaf' : range(1,11,5), 'n_estimators': range(20,100,40), 'max_depth': range(5,25,10)}
    gb_hparams = { 'loss' : ['squared_error', 'absolute_error'], 'learning_rate' : [(0.1 * 10 ** x) for x in range(0, 4)]}
    nn_hparams = { 
        'max_epochs' : [25], 
        'lr': [1/(10**x) for x in range(1,4)], 
        'optimizer': [optim.SGD], #[ optim.SGD, optim.Adam ], 
        'criterion': [MSELoss], 
        # We don't have a lot of data here, batching may be faster but we lose information in the process - stick w/ 1
        'batch_size': [1],  
        # Cue the torch DataLoader to shuffle training data
        'iterator_train__shuffle': [True], 
        'module__n_input' : [nn_input_size], 
        'module__n_hidden1': range(50,100,25), 
        'module__n_hidden2': range(0,50,25),
        'module__n_hidden2': range(0,10,5),
        # GridSearch implements cross validation, disable skorch internal holdout 
        'train_split': [None], 
        # Leverage cuda device if we find it/them
        'device': ['cuda' if gpu_survey() else 'cpu'], 
        }

    # TODO: introduce PCA here, regardless of whether it's running at a higher level to find optimal models  
    # Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=12)), ('grid', GridSearchCV(?, ?, error_score=-1))]),

    experiments = [
        #Pipeline([('model', DummyRegressor())]), # Control
        Pipeline([('nn', NeuralNetRegressor(module=WARNet, criterion=MSELoss, optimizer=SGD, max_epochs=100, lr=0.0001, module__n_input=nn_input_size, module__n_hidden1=100, batch_size=1, iterator_train__shuffle=True))]),
        #Pipeline([('grid', GridSearchCV(Ridge(), lr_hparams, n_jobs=-1, error_score=-1))]), 
        #Pipeline([('poly', PolynomialFeatures()), ('model', LinearRegression())]),
        #Pipeline([('pca', PCA(n_components=5)), ('model', LinearRegression())]),
        #Pipeline([('grid', GridSearchCV(LinearRegression(), {}, n_jobs=-1, refit=True, error_score=-1))]),
        #Pipeline([('grid', GridSearchCV(Lasso(), lr_hparams, n_jobs=-1, refit=True, error_score=-1))]), 
        #Pipeline([('scaler', StandardScaler()), ('grid', GridSearchCV(SVR(), sv_hparams, refit=True, n_jobs=-1, error_score=-1))]),
        #Pipeline([('scaler', StandardScaler()), ('poly', PolynomialFeatures()), ('grid', GridSearchCV(SVR(), sv_hparams, refit=True, n_jobs=-1, error_score=-1))]),
        #Pipeline([('grid', GridSearchCV(RandomForestRegressor(), rf_hparams, refit=True, n_jobs=-1,error_score=-1))]),
        #Pipeline([('grid', GridSearchCV(GradientBoostingRegressor(), gb_hparams, refit=True, n_jobs=-1,error_score=-1))]),
        
        #Pipeline([('nn', NeuralNetRegressor(module=WARNet, criterion=MSELoss, optimizer=SGD, max_epochs=30, lr=0.5, module__n_input=nn_input_size, module__n_hidden1=10, batch_size=1, iterator_train__shuffle=True))]),
        #Pipeline(['nn', NeuralNetRegressor(module=WARNet, criterion=MSELoss, optimizer=SGD, max_epochs=15, lr=0.1, module__n_input=nn_input_size, module__n_hidden1=10, iterator_train__shuffle=True))]),
        Pipeline([('gridnn', 
                   GridSearchCV(
                       NeuralNetRegressor(module=WARNet), 
                       nn_hparams, 
                       refit=True, 
                       n_jobs=-1,
                       error_score=-1)
            )]),
        # TODO: addPCA transform before NN estimator here to
    ]

    return experiments

def search(X_train, y_train, splits=3, visualize=False):  
    """
    Perform a hyperparameter and model search across all promising algorithms
    """

    min_loss = 0.5
    experiments = generate_experiments(nn_input_size=len(X_train.columns))
    candidates = evaluate(experiments, X_train, y_train, min_loss, False)

    # Find a winning experiment - note there is some redundancy here given the test/train split 
    # done implicitly in all of our parameter search efforts in the experiment 'manifest'. However, 
    # as we discover leading candidates, we may pull them out of the grid search and memorialize them 
    # to ensure they are perpetually considered. In these cases it's essential for the below validation 
    # step, lest we accidentally nominate models above that hasn't undergone recent cross validation. 
    winner, mse = validate(X_train, y_train, candidates, visualize, splits)
    if winner is not None: 
        print(f"Best model identified: {get_estimator_from_experiment(winner)} (mse: {mse}).")
    else: 
        print(f"No model performance exceeded provided threshold ({min_loss})! ")

    return winner, mse

def monolithic_search(splits=3, visualize=False): 
    """
    Look for a single algorithm to maximize performance across all features
    """
    X_train, y_train, X_test, y_test = build_train_test_set(standard=True, statcast=True) 
    winner, mae = search(X_train, y_train, splits=splits, visualize=visualize)
    
    # TODO: predict on the test set and check performance! below needs to be refactored 
    return winner, mae

def cluster_search(dimensions=2, splits=3, visualize=False):  
    """
    Perform a cluster-based hyperparameter and model search across all promising algorithms, 
    using the resulting ensemble to predict classes 

    NOTE: this is a refactored grid search pipeline used in the kaggle comp
    TODO: ensure this can reduce down to the monolithic search by requesting a single cluster
    """    
    X_train, y_train, X_test, y_test = build_train_test_set(standard=True, statcast=False) 
    
    # Leverage the low-dimensional feature and associated label to isolate prominent sub-distributions 
    # and fit an experiment(pipeline) tailored to classify each
    
    # TODO complete dimensionality reduction and clustering 
    if dimensions <= 1: 
        raise ValueError()
    
    train_Xstd['cluster'] = [1] * len(train_Xstd)
    
    cluster_winners = { }  
    for cluster in X_train['cluster'].unique(): 
        
        X_train_clust = X_train[X_train['cluster'] == cluster]
        y_train_clust = pd.DataFrame(y_train).join(X_train_clust, how='inner')['label']

        # Where our clusters consist of uniform labels, short-circuit the search, the best strategy is 
        # to predict this label for any input
        winner, mse = search(X_train_clust, y_train_clust, splits=splits, visualize=visualize)
        print(f"Best model identified for cluster {cluster}: {get_estimator_from_experiment(winner)} (MSE of {mse}).")

        cluster_winners[cluster] = winner

    return cluster_winners 

def main(**args): 
    """
    CLI entry point and arg handler
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-d", "--dimensions")
    parser.add_argument("-k", "--splits")
    parser.add_argument("-c", "--conventional", action=argparse.BooleanOptionalAction)
    parser.add_argument("-s", "--statcast", action=argparse.BooleanOptionalAction)
    parser.add_argument("-v", "--visualize", action=argparse.BooleanOptionalAction)
    
    parser.set_defaults(visualize=False)
    args = parser.parse_args()
    
    monolithic_search()
    #cluster_search(int(args.dimensions), args.conventional, args.statcast, int(args.splits), args.visualize)

if __name__ == "__main__": 
    main()