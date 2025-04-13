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
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from skorch import NeuralNetRegressor 
from data import load_statcast, load_standard
from nn import WARNet

def scale(df, range=(0,1), omit=[]):
    """
    Scale a column into a given range, omitting one or more columns if provided 
    
    NOTE: utilty function written for the Kaggle comp
    """
    for column in df.columns: 
        if column not in omit: 
            df[column] = min_max_scale(df, column, (0,1))

    return df

def min_max_scale(df, column, range=(0,1)): 
    """
    Scale a column in a DF to the provided range     

    NOTE: utilty written for the Kaggle comp
    """
    scaler = MinMaxScaler(feature_range=range)
    scaled = scaler.fit_transform(df[[column]]) 
    return scaled.transpose()[0]

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

def get_model_from_experiment(experiment):
    """
    Retrieve the model from a fit pipeline (we use a specific tag to ID them)

    NOTE: repurposed from kaggle comp
    """
    if 'model' in experiment.named_steps.keys():
        model = experiment.named_steps['model']
    elif 'grid' in experiment.named_steps.keys(): 
        model = experiment.named_steps['grid'].best_estimator_
    else: 
        raise ValueError("Cannot extract estimator from pipeline!")

    return model 

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
        X_train.join(sc23, how="inner", inplace=True)
        X_test = std24.drop(['WAR'], axis=1)
        X_test.join(sc24, how="inner", inplace=True)
    
    return X_train, y_train, X_test, y_test

@ignore_warnings(category=ConvergenceWarning)
def evaluate(experiments, X_train, y_train, threshold=0.2, visualize=False): 
    """
    Search for optimal models. No internal cross validation is performed unless it's
    implicit in the provided experiment pipeline execution. Ensure validation happens outside 
    of this process for models that don't incorporate validation into the associated pipeline.

    Returns the experiments (sklearn pipelines) whose error falls below the provided threshold
    """
    winners = []

    for i, experiment in enumerate(experiments):         
        experiment.fit(X_train, y_train)
        
        preds = experiment.predict(X_train)

        mse = metrics.mean_squared_error(y_train, preds)

        print(f"==== \nExperiment {i}: {mse}\n")

        if mse < threshold: 
            winners.append(experiment)
    
        if visualize: 
            # TODO: fix this  
            viz_df, centroids = project_results_2d(X_train, y_train, preds, threshold=0.5, clusters=5)
            visualize_results_2d(viz_df, centroids, title=f"Pipeline {i}, Model: {str(get_model_from_experiment(experiment))}", c_filter=[])

    return winners

def validate(X_train, y_train, candidates, visualize=False, splits=5): 
    """
    Validate the candidate experiments (sklearn pipelines) with cross validation
    and return a winner. 
    """
    kf = StratifiedKFold(n_splits=splits, shuffle=False)

    winner_mae = None
    winner = None
    
    for candidate in candidates: 

        mae = None
        for train_ix, test_ix in kf.split(X_train, y_train):            
            candidate.fit(X_train.iloc[train_ix], y_train.iloc[train_ix])
        
            preds = candidate.predict(X_train.iloc[test_ix])

            fold_mae = metrics.mean_squared_error(y_train.iloc[test_ix], preds)
            mae = (mae if mae is not None else 0) + fold_mae/splits

        print(f"======== \nCandidate {get_model_from_experiment(candidate)}: {mae}\n")
        
        if visualize: 
            # TODO: fix - this isn't classification based and the PCA support functions were refactored
            preds = candidate.predict(X_train)

            viz_df, centroids = project_results_2d(X_train, y_train, preds, threshold=0.5, clusters=5)
            visualize_results_2d(viz_df, centroids, title=f"Model: {str(get_model_from_experiment(candidate))} @ {mae}", c_filter=None)
        
        if winner_mae > mae: 
            winner_mae = mae 
            winner = candidate 
    
    return winner, winner_mae

lr_hparams = { 'alpha' : [2**x for x in range(0,10,1) ] }
sv_hparams = { 'C' : [0.2, 0.4], 'kernel' : ['poly', 'rbf' ] }
rf_hparams = { 'min_samples_leaf' : range(1,11,5), 'n_estimators': range(20,100,40), 'max_depth': range(5,25,10)}
gb_hparams = { 'loss' : ['squared_error', 'absolute_error'], 'learning_rate' : [(0.1 * 10 ** x) for x in range(0, 4)]}
#TODO: YIKES ... skorch seemed like the ticket, but how do we prep our data and fire it into the NN if we use gridsearch/skorch? 
nn_hparams = { 'max_epochs' : range(5,20,5), 'lr': [0.1, 0.2, 0.3], 'module__n_input': 1000, 'module__n_hidden1': range(10,100,15), 'module__n_hidden2': [5, 10] }

# Our experiment register. This will grow and shrink as experiments are run and models and the 
# preferred hyperparameters are identified. 
#
# NOTE estimators must be tagged w/ 'model' or 'grid' for GridSearch to enable recovery of 
# optimal models for post-evaluation re-training 
# TODO: introduce PCA here, regardless of whether it's running at a higher level to find optimal models  
#Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=12)), ('grid', GridSearchCV(?, ?, error_score=-1))]),
experiments = [
    Pipeline([('model', DummyRegressor())]), # Control
    Pipeline([('grid', GridSearchCV(LinearRegression(), {}, n_jobs=-1, error_score=-1))]),
    Pipeline([('grid', GridSearchCV(Ridge(), lr_hparams, n_jobs=-1, error_score=-1))]), # best: alpha = 2
    Pipeline([('grid', GridSearchCV(Lasso(), lr_hparams, n_jobs=-1, error_score=-1))]), 
    Pipeline([('poly', PolynomialFeatures()), ('grid', GridSearchCV(LinearRegression(), {}, n_jobs=-1, error_score=-1))]), # ! bestest!
    Pipeline([('scaler', StandardScaler()), ('grid', GridSearchCV(SVR(), sv_hparams, n_jobs=-1, error_score=-1))]),
    Pipeline([('scaler', StandardScaler()), ('poly', PolynomialFeatures()), ('grid', GridSearchCV(SVR(), sv_hparams, n_jobs=-1, error_score=-1))]),
    Pipeline([('grid', GridSearchCV(RandomForestRegressor(), rf_hparams, n_jobs=-1,error_score=-1))]),
    Pipeline([('grid', GridSearchCV(GradientBoostingRegressor(), gb_hparams, n_jobs=-1,error_score=-1))]),
    Pipeline([('model', NeuralNetRegressor(WARNet, max_epochs=10, lr=0.1, iterator_train__shuffle=True))]),
    Pipeline([('model', GridSearchCV(NeuralNetRegressor(module=WARNet), nn_hparams, n_jobs=-1,error_score=-1))]),
]

def search(X_train, y_train, splits=3, visualize=False):  
    """
    Perform a hyperparameter and model search across all promising algorithms
    """
    global experiments 

    candidates = evaluate(experiments, X_train, y_train, 0.5, False)

    # Find a winning experiment - note there is some redundancy here given the test/train split 
    # done implicitly in all of our parameter search efforts in the experiment 'manifest'. However, 
    # as we discover leading candidates, we may pull them out of the grid search and memorialize them 
    # to ensure they are perpetually considered. In these cases it's essential for the below validation 
    # step, lest we accidentally nominate models above that hasn't undergone recent cross validation. 
    winner, mse = validate(X_train, y_train, candidates, visualize, splits)
    print(f"Best model identified: {get_model_from_experiment(winner)} (mse: {mse}).")

    return winner, mse

def monolithic_search(splits=3, visualize=False): 
    """
    Look for a single algorithm to maximize performance across all features
    """
    X_train, y_train, X_test, y_test = build_train_test_set(standard=True, statcast=False) 
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
        print(f"Best model identified for cluster {cluster}: {get_model_from_experiment(winner)} (MSE of {mse}).")

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