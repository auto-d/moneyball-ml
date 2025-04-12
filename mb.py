import re 
import argparse
from datetime import datetime
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn import metrics
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification

from data import load_statcast, load_standard

# TODO: these have all been rewritten, test

def pca(df, components=2): 
    """
    Compress the provided dataframe into n dimensions, returning the new dims 
    as columsn in a D
    
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

def categorize_prediction(label, pred): 
    """
    What the name says 

    NOTE: this is a utilty function I wrote for the Kaggle comp
    """
    if label and pred: 
        return "TP"
    elif label and not pred: 
        return "FN"
    elif not label and pred: 
        return "FP"
    elif not label and not pred: 
        return "TN"

# TODO: rewrite with the new methods above
def project_results_2d(X_train, y_train, probs, threshold=0.5, clusters=8): 
    """
    Use PCA to prepare a flattened version of the training data and our performance on the TRAINING data predictions

    NOTE: this is a utilty function I wrote for the Kaggle comp
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

    NOTE: this is a utilty function I wrote for the Kaggle comp
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

@ignore_warnings(category=ConvergenceWarning)
def bakeoff(experiments, X_train, y_train, threshold=0.9, visualize=False): 
    """
    Iterate over various options, looking for an optimal model given the data. This 
    function expects all cross-validation to happen within the model pipeline. 

    Returns the experiments (sklearn pipelines) whose AUROC exceeded the provided threshold. 

    NOTE: repurposed from kaggle comp
    """
    winners = []    

    for i, experiment in enumerate(experiments):         
        experiment.fit(X_train, y_train)
        
        probs = None 
        if hasattr(experiment, 'predict_proba'): 
            probs = experiment.predict_proba(X_train)[:,1]
        elif hasattr(experiment, 'predict'):
            probs = experiment.predict(X_train)

        roc = metrics.roc_auc_score(y_train, probs)

        print(f"==== \nExperiment {i}: {roc}\n")

        if roc > threshold: 
            winners.append(experiment)
    
        if visualize: 
            viz_df, centroids = project_results_2d(X_train, y_train, probs, threshold=0.5, clusters=5)
            visualize_results_2d(viz_df, centroids, title=f"Pipeline {i}, Model: {str(get_model_from_experiment(experiment))}", c_filter=[])

    return winners

def build_train_set():
    """
    Load, transform, and apply engineered features, returning the training set
    """
    
    # Create training sets for standard and statcast data, holding WAR out as a label
    std = load_standard(2023)
    train_y = std['WAR']
    train_Xstd = std.drop('WAR', axis=1)
    train_Xsc = load_statcast(2023)    
    
    # Repeat for test set
    std = load_standard(2024)
    test_y = std['WAR']
    test_Xstd = std.drop(['WAR'], axis=1)
    test_Xsc = load_statcast(2024)
    
    return train_Xstd, train_Xsc, train_y, test_Xstd, test_Xsc, test_y

def validate(X_train, y_train, candidates, visualize=False, splits=5): 
    """
    Validate the candidate experiments with cross validation and return a winner

    NOTE: repurposed from kaggle comp
    """
    kf = StratifiedKFold(n_splits=splits, shuffle=False)

    winner_roc = 0
    winner = None
    
    # Iterate over the candidate models and evalute on k folds of the data
    for candidate in candidates: 

        roc = 0 
        for train_ix, test_ix in kf.split(X_train, y_train):            
            candidate.fit(X_train.iloc[train_ix], y_train.iloc[train_ix])
        
            probs = None 
            if hasattr(candidate, 'predict_proba'): 
                probs = candidate.predict_proba(X_train.iloc[test_ix])[:,1]
            elif hasattr(candidate, 'predict'):
                probs = candidate.predict(X_train.iloc[test_ix])

            fold_roc = metrics.roc_auc_score(y_train.iloc[test_ix], probs)
            roc += fold_roc/splits

        # Retrain in preparation for visualization or submission
        print(f"======== \nCandidate {get_model_from_experiment(candidate)}: {roc}\n")
        candidate.fit(X_train, y_train)
        
        # If we're going to plot, we need updated predictions
        if visualize: 
            probs = None 
            if hasattr(candidate, 'predict_proba'): 
                probs = candidate.predict_proba(X_train)[:,1]
            elif hasattr(candidate, 'predict'):
                probs = candidate.predict(X_train)

            viz_df, centroids = project_results_2d(X_train, y_train, probs, threshold=0.5, clusters=5)
            visualize_results_2d(viz_df, centroids, title=f"Model: {str(get_model_from_experiment(candidate))} @ {roc}", c_filter=None)
        
        if winner_roc < roc: 
            winner_roc = roc 
            winner = candidate 
    
    return winner, winner_roc

lr_hparams = { 'penalty' : ('l1', 'l2'), 'C' : [x / 10 for x in range(1,4)]}
rf_hparams = { 'min_samples_leaf' : range(3,5,1), 'n_estimators': range(40,50,5), 'max_depth': range(7,9,1)}
sv_hparams = { 'C' : [x / 10 for x in range(1,5,2)], 'kernel' : ['sigmoid', 'rbf'] }
kn_hparams = { 'n_neighbors' : range(7,9)}
experiments = [
    # NOTE estimator must be tagged w/ 'model' for or 'grid' for GridSearch to enable predictions

    # Control
    #Pipeline([('model', DummyClassifier())]),

    # Experimental models and hyperparameter tuning stuff -- these should all be gridsearch estimators, as
    # we are delegating cross-validation to that object and relying on the ability to retrieve the optimal model 
    #Pipeline([('scaler', StandardScaler()), ('poly', PolynomialFeatures()), ('grid', GridSearchCV(SVC(), sv_hparams, error_score=0))]),
    #Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=12)), ('grid', GridSearchCV(SVC(), sv_hparams, error_score=0))]),

    #Pipeline([('scaler', StandardScaler()), ('poly', PolynomialFeatures()), ('grid', GridSearchCV(KNeighborsClassifier(), kn_hparams, error_score=0))]), 
    
    # Logistic Regression w/ L1/L2 norm penalties         
    Pipeline([('grid', GridSearchCV(LogisticRegression(),lr_hparams, scoring='roc_auc', error_score=0))]),

    # Random forest         
    #Pipeline([('grid', GridSearchCV(RandomForestClassifier(),rf_hparams, scoring='roc_auc', error_score=0))]),
    
    # PCA + LR
    Pipeline([('scaler', StandardScaler()), ('pca3', PCA(n_components=12)), ('model', SVC(kernel='rbf'))]),
    # Pipeline([('scaler', StandardScaler()), ('pca3', PCA(n_components=5)), ('model', SVC(kernel='rbf'))]),
    Pipeline([('scaler', StandardScaler()), ('pca3', PCA(n_components=12)), ('model', SVC(kernel='sigmoid'))]),
    # Pipeline([('scaler', StandardScaler()), ('pca6', PCA(n_components=26)), ('model', LogisticRegression(penalty=None))]),
    # Pipeline([('scaler', StandardScaler()), ('pca8', PCA(n_components=8)), ('model', LogisticRegression(penalty=None))]),

    # Best performing models and hyperparameters reseulting from above search operations, we'll cross validate again 
    # before submissiont to find the best based on the current features
    Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=7)), ('model', LogisticRegression(penalty=None))]),
    Pipeline([('scaler', StandardScaler()), ('pca4', PCA(n_components=34)), ('model', LogisticRegression(penalty=None))]),
    Pipeline([('model', RandomForestClassifier(max_depth=8, min_samples_leaf=4, n_estimators=45))]),
    Pipeline([('scaler', StandardScaler()), ('model', KNeighborsClassifier(n_neighbors=8))]), 
    # Pipeline([('model', VotingClassifier(
    #     estimators=[
    #         ('RF', RandomForestClassifier(max_depth=9, min_samples_leaf=4, n_estimators=40)), 
    #         ('KNN', Pipeline([('scaler', StandardScaler()), ('model', KNeighborsClassifier(n_neighbors=8))])), 
    #         ],
    #     voting='soft'))]),
    Pipeline([('model', VotingClassifier(
        estimators=[
            ('RF', RandomForestClassifier(max_depth=8, min_samples_leaf=4, n_estimators=70)), 
            ('KNN', Pipeline([('scaler', StandardScaler()), ('model', KNeighborsClassifier(n_neighbors=8))])), 
            ],
        voting='soft'))]),
    # This performs best on internal validation, but is a few tenths below the above voting classifier on the hidden 
    # test data. Keep it here for potential mutation. 
    # Pipeline([('model', VotingClassifier(
    #     estimators=[
    #         ('RF', RandomForestClassifier(max_depth=8, min_samples_leaf=4, n_estimators=45)), 
    #         ('KNN', Pipeline([('scaler', StandardScaler()), ('model', KNeighborsClassifier(n_neighbors=8))])), 
    #         ('LR', Pipeline([('scaler', StandardScaler()), ('pca4', PCA(n_components=34)), ('model', LogisticRegression(penalty=None))])),
    #         ],
    #     voting='soft'))]),
]

def search(X_train, y_train, splits=3, visualize=False):  
    """
    Perform a hyperparameter and model search across all promising algorithms

    NOTE: repurposed from kaggle comp
    """
    global experiments 

    # Check algorithm outcomes on enriched account data
    candidates = bakeoff(experiments, X_train, y_train, 0.85, False)

    # Find a winning experiment
    winner, roc = validate(X_train, y_train, candidates, visualize, splits)
    print(f"Best model identified: {get_model_from_experiment(winner)} (AUROC of {roc}).")

    return winner, roc

def cluster_search(splits=3, visualize=False):  
    """
    Perform a cluster-based hyperparameter and model search across all promising algorithms, 
    using the resulting ensemble to predict classes 

    NOTE: repurposed from kaggle comp
    TODO: ensure this can reduce down to the monolithic search by requesting a single cluster
    """    
    X_train, y_train, X_test = build_train_set()
    
    # Leverage the low-dimensional feature and associated label to isolate prominent sub-distributions 
    # and fit an experiment(pipeline) tailored to classify each
    cluster_winners = { }  
    for cluster in X_train['cluster'].unique(): 
        
        X_train_clust = X_train[X_train['cluster'] == cluster]
        y_train_clust = pd.DataFrame(y_train).join(X_train_clust, how='inner')['label']

        # Where our clusters consist of uniform labels, short-circuit the search, the best strategy is 
        # to predict this label for any input
        winner = None 
        if len(y_train_clust.unique()) != 1: 
            winner, roc = search(X_train_clust, y_train_clust, splits=splits, visualize=visualize)
            print(f"Best model identified for cluster {cluster}: {get_model_from_experiment(winner)} (AUROC of {roc}).")
        else:
            winner = Pipeline([('model', DummyClassifier(strategy='most_frequent'))]).fit(X_train_clust, y_train_clust)
            print(f"Bypassed model selection for cluster {cluster} due to uniform labels ({y_train_clust[0]}) falling back to {get_model_from_experiment(winner)}.")

        cluster_winners[cluster] = winner

def main(**args): 
    """
    CLI entry point and arg handler
    """
    parser = argparse.ArgumentParser()
    
    #group = parser.add_mutually_exclusive_group() 
    #group.add_argument("-s", "--search", action=argparse.BooleanOptionalAction)
    #group.add_argument("-c", "--clustersearch", action=argparse.BooleanOptionalAction)

    parser.add_argument("-k", "--splits")
    parser.add_argument("-v", "--visualize", action=argparse.BooleanOptionalAction)
    
    parser.set_defaults(visualize=False)
    args = parser.parse_args()
    
    build_train_set()
    #cluster_search(int(args.splits))
    
    #parser.print_help()

if __name__ == "__main__": 
    main()