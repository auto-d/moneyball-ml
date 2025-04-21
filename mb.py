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
from data import load_statcast, load_standard, find_na, find_player
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
    df2 = df.copy()     
    for column in df2.columns: 
        if column not in omit: 
            df2[column] = min_max_scale(df2, column, (0,1))

    return df2

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

def build_train_test_set(standard=True, statcast=True):
    """
    Load, transform, and apply engineered features, returning the training set. 
    """

    if not standard and not statcast: 
        raise ValueError("One of standard or statcast must be True")
    
    X_train = pd.DataFrame()

    # Create training sets for standard and statcast data, holding WAR out as a label
    # regardless of which dataset(s) will be used
    std22 = load_standard(2022)
    std23 = load_standard(2023)
    std24 = load_standard(2024)

    y_train = std23['WAR']
    y_test = std24['WAR']    

    sc22 = load_statcast(2022) if statcast else None
    sc23 = load_statcast(2023) if statcast else None
    
    if standard and not statcast:
        X_train = std22.drop(['WAR'], axis=1)
        X_test = std23.drop(['WAR'], axis=1)
    elif statcast and not standard: 
        X_train = sc22
        X_test = sc23
    else: 
        X_train = std22.drop(['WAR'], axis=1)
        X_train = X_train.join(sc22, how="left")
        X_test = std23.drop(['WAR'], axis=1)
        X_test = X_test.join(sc23, how="left")

    # We can only train on players for which we have a label, intersect our training data 
    # with the player IDs from the labelset. Similar logic for the test set.
    X_train = X_train[X_train.index.isin(y_train.index)]
    y_train = y_train[y_train.index.isin(X_train.index)]

    X_test = X_test[X_test.index.isin(y_test.index)]   
    y_test = y_test[y_test.index.isin(X_test.index)]
    
    if len(X_train) != len(y_train) or len(X_test) != len(y_test) or (len(X_train.columns) != len(X_test.columns)):
        raise ValueError("Inconsistent data and label lengths, aborting run!")

    X_train.fillna(0., inplace=True)
    X_test.fillna(0., inplace=True)

    # Canary 
    find_na(X_train)
    find_na(X_test)

    return X_train, y_train, X_test, y_test

def make_nn_set(X, y=None, scaler=None): 
    """
    Transform to skorch/torch-compatible inputs, scale prior if a fit scaler if furnished
    """

    Xnn = pd.DataFrame(scaler.transform(X)) if scaler else X 
    Xnn = Xnn.to_numpy().astype(np.float32)

    ynn = None 
    if y is not None:
        ynn = y.to_numpy().astype(np.float32)
        ynn = np.expand_dims(ynn, axis=1)

    return Xnn, ynn

@ignore_warnings(category=ConvergenceWarning)
def run_experiments(experiments, X_train, y_train, threshold=0.5): 
    """
    Search for optimal models. No internal cross validation is performed, it's expected to be
    implicit in the provided experiment pipeline execution. Winning configurations will be 
    cross validated explicitly when app runs in evaluation mode. 

    Returns the experiments (sklearn pipelines) whose error falls below the provided threshold
    """
    winners = []

    scaler = StandardScaler().fit(X_train)
    X_train_nn, y_train_nn = make_nn_set(X_train, y_train, scaler)

    for i, experiment in enumerate(experiments):         

        print(f"===================================\n")
        print(f"Experiment {i}:\n")
        
        # If the estimator in the experiment is a neural network, use a skorch-compatible format. 
        # Note we're relying on gridsearch to select the optimal model config w/ cross validation. 
        # We then do a lazy test on the TRAINING data here to log the score on the whole set. These
        # will be cross validated again before selection and moving on to face any test data. Note 
        # gridsearch implicitly (by default anyway) refits on the best configuration identified, so
        # we're only getting the output of the best model here when we predict. 
        if is_nn_experiment(experiment): 
            X = X_train_nn
            y = y_train_nn
        else: 
            X = X_train 
            y = y_train
        
        experiment.fit(X, y)
        preds = experiment.predict(X)

        mse = metrics.mean_squared_error(y_train, preds)

        grid_name = 'gridnn' if is_nn_experiment(experiment) else 'grid'

        print(f"⚙️ {experiment}/{i}:")
        print(f"⚡️ best estimator : {experiment.named_steps[grid_name].best_estimator_}")
        print(f"⚡️ best params : {experiment.named_steps[grid_name].best_params_}")
        print(f"⚡️ best score : {experiment.named_steps[grid_name].best_score_}")
        print(f"⚡️ MSE: {mse}")
        print(f"-----------------------------------\n")

        if mse < threshold: 
            winners.append(experiment)
            
    return winners

def validate(X_train, y_train, candidates, splits=5): 
    """
    Cross-validate the candidate experiments (sklearn pipelines) and return a winner. 
    """
    kf = KFold(n_splits=splits, shuffle=False)

    winner_mse = None
    winner = None
    
    # skorch will eat a torch DataSet if passed directly to model.fit, but sklearn's transformations 
    # have no idea how to deal with a DataSet object. This presents a problem for transformations in the 
    # experiment paradigm used elsewhere here. We should write an sklearn transformer that can handle the 
    # DataSet, but since our only transformation is scaling, just apply it manually here (external to the 
    # sklearn experimenet pipeline) as well as in the evaluation logic above (see run_experiments())
    X_train_scaled = pd.DataFrame(StandardScaler().fit_transform(X_train)) 
    
    for candidate in candidates: 

        mse = None
        for train_ix, test_ix in kf.split(X_train, y_train):            
            
            preds = None 

            if is_nn_experiment(candidate):                     
                X, y = make_nn_set(X_train_scaled.iloc[train_ix], y_train.iloc[train_ix])
                X_val, _ = make_nn_set(X_train_scaled.iloc[test_ix])
                
            else: 
                X = X_train.iloc[train_ix]
                y = y_train.iloc[train_ix]
                X_val = X_train.iloc[test_ix]

            candidate.fit(X, y) 
            preds = candidate.predict(X_val)

            fold_mse = metrics.mean_squared_error(y_train.iloc[test_ix], preds)
            mse = (mse if mse is not None else 0) + fold_mse/splits

        print(f"======== \nCandidate {get_estimator_from_experiment(candidate)}: {mse}\n")

        if (winner_mse is None) or (winner_mse > mse):
            winner_mse = mse 
            winner = candidate 
    
    # Retrain the winning pipeline on the full dataset
    if is_nn_experiment(winner): 
        X, y = make_nn_set(X_train_scaled, y_train)
    else: 
        X = X_train
        y = y_train

    winner.fit(X, y)
    
    return winner, winner_mse

def is_nn_experiment(experiment): 
    """
    What name says
    """
    keys = experiment.named_steps.keys()
    return True if 'nn' in keys or 'gridnn' in keys else False

def is_grid_experiment(experiment): 
    """
    What name says
    """
    keys = experiment.named_steps.keys()
    return True if 'grid' in keys or 'gridnn' in keys else False

def get_estimator_from_experiment(experiment):
    """
    Retrieve the model or best estimator from a fit pipeline (we use a specific tag to ID them)

    NOTE: repurposed from kaggle comp
    """
    keys = experiment.named_steps.keys()

    if 'model' in keys:
        model = experiment.named_steps['model']
    elif 'nn' in keys:
        model = experiment.named_steps['nn']        
    elif 'grid' in keys: 
        model = experiment.named_steps['grid'].best_estimator_
    elif 'gridnn' in keys: 
        model = experiment.named_steps['gridnn'].best_estimator_
    else: 
        raise ValueError("Cannot extract estimator from pipeline!")

    return model 

def build_experiments(nn_input_size):
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

    lr_hparams = { 'alpha' : [(0.0001 * 10 ** x) for x in range(0,4,1) ] }
    sv_hparams = { 'C' : [0.3, 0.4, 0.5], 'kernel' : ['rbf' ] }
    rf_hparams = { 'min_samples_leaf' : range(3,8,1), 'n_estimators': range(10,30,5), 'max_depth': range(10,20,2)}
    gb_hparams = { 'loss' : 'squared_error', 'learning_rate' : [(0.0001 * 10 ** x) for x in range(0, 4)]}
    nn_hparams = { 
        'max_epochs' : [200], 
        'lr': [1/(10**x) for x in range(2,4)],
        'optimizer': [optim.SGD], #[ optim.SGD, optim.Adam ], 
        'criterion': [MSELoss], 
        # We don't have a lot of data here, batching may be faster but we lose information in the process - stick w/ 1
        'batch_size': [5,7], 
        # Cue the torch DataLoader to shuffle training data
        'iterator_train__shuffle': [True], 
        'module__n_input' : [nn_input_size], 
        'module__n_hidden1': range(80,120,20), 
        'module__n_hidden2': range(0,25,10),
        #'module__n_hidden3': range(0,10,5),
        # GridSearch implements cross validation, disable skorch internal holdout 
        'train_split': [None], 
        # Leverage cuda device if we find it/them
        'device': ['cuda' if gpu_survey() else 'cpu'], 
        }

    experiments = [    
        #Pipeline([('grid', GridSearchCV(Ridge(), lr_hparams, scoring='neg_mean_squared_error', n_jobs=-1, error_score=-1))]), 
        Pipeline([('gridnn', GridSearchCV(NeuralNetRegressor(module=WARNet, verbose=0), nn_hparams, scoring='neg_mean_squared_error', n_jobs=-1,error_score=-1))]), 
        # Pipeline([('grid', GridSearchCV(Lasso(), lr_hparams, scoring='neg_mean_squared_error', n_jobs=-1, error_score=-1))]), 
        # Pipeline([('scaler', StandardScaler()), ('grid', GridSearchCV(SVR(), sv_hparams, scoring='neg_mean_squared_error', n_jobs=-1, error_score=-1))]),
        # Pipeline([('grid', GridSearchCV(RandomForestRegressor(), rf_hparams, scoring='neg_mean_squared_error', n_jobs=-1, error_score=-1))]),
        # Pipeline([('grid', GridSearchCV(GradientBoostingRegressor(), gb_hparams, scoring='neg_mean_squared_error', n_jobs=-1,error_score=-1))]),
    ]

    return experiments

def build_candidates(nn_input_size): 
    """
    Construct a list of candidate model pipelines and return. These are manually mapped over 
    from the experimentation phase to clarify the cross-validation logic, though they should lose any 
    HP search classes like GridSearch in the process. We'll do our own CV here and should only have
    optimal configurations represented. 
    """
    candidates = [
        Pipeline([('model', DummyRegressor())]), # Control
        Pipeline([('model', Lasso(alpha=0.1))]), 
        Pipeline([('nn', NeuralNetRegressor(
            module=WARNet, 
            criterion=MSELoss, 
            optimizer=SGD, 
            max_epochs=120, 
            lr=0.001, 
            module__n_input=nn_input_size, 
            module__n_hidden1=100, 
            batch_size=7, 
            iterator_train__shuffle=True, 
            verbose=0))]),
        Pipeline([('model', RandomForestRegressor(max_depth=85, min_samples_leaf=6, n_estimators=25))]),
        Pipeline([('model', GradientBoostingRegressor(learning_rate=0.1))]),
    ]

    return candidates

def search(X_train, y_train, splits=3, threshold=0.3, visualize=False):  
    """
    Perform a hyperparameter and model search across all promising algorithms using 
    exclusively the training data. Identifies all model pipelines that report an error 
    lower than the provided threshold. Returns the optimal pipeline fit on the training data. 
    """    
    experiments = build_experiments(nn_input_size=len(X_train.columns))
    candidates = run_experiments(experiments, X_train, y_train, threshold)

    print(f"Algorithm search identified: {len(candidates)} below {threshold} MSE:")

def build_results(y_test, preds): 
    """
    Assemble a results dataframe to simplify analysis
    """
    errors = (y_test.to_numpy() - preds)
    results = pd.DataFrame() 
    results['player_id'] = y_test.index
    results['war'] = y_test.to_numpy()
    results['war_p'] = preds
    results['error'] = errors
    results['player_first'] = results['player_id'].apply(lambda x: find_player(id=[int(x)])['name_first'])
    results['player_last'] = results['player_id'].apply(lambda x: find_player(id=[int(x)])['name_last'])
    results.sort_values(by='error', inplace=True)
    
    return results

def heatmap_results(results): 
    """
    Plot our errors vs true with density (color). 
    """
    fig, ax = plt.subplots(figsize=(10,8))
    hb = ax.hexbin(x=results.error, y=results.war, gridsize=40, cmap='inferno')
    
    # We have a few outliers that are barely perceptable in the hexbin plot 
    # reduce the max value here so we can better appreciate the density of the majority 
    # of values
    xlim = results.error.min(), results.error.max()-3
    ylim = results.war.min(), results.war.max()-3
    ax.set(xlim=xlim, ylim=ylim)    
    ax.set_xlabel("2024 WAR")
    ax.set_ylabel("Prediction Error") 
    ax.set_title('Target vs Prediction Error')
    cb = fig.colorbar(hb, ax=ax, label='counts')
    plt.show()

def show_error_distribution(results): 
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.hist(results.error, bins=100)
    ax.set_xlabel('Error')
    ax.set_xticks(range(-8,9,1))
    ax.set_ylabel('Count')
    ax.set_title("Prediction Error Distribution")

def evaluate(X_train, y_train, X_test, y_test, splits=5, visualize=False): 
    """
    Perform cross-validation on our candidate model pipelines and apply the winner to 
    the test challenge 
    """
    candidates = build_candidates(nn_input_size=len(X_train.columns))
    winner, mse = validate(X_train, y_train, candidates, splits)
    if winner is not None: 
        print(f"Best model pipeline identified: {get_estimator_from_experiment(winner)} (mse: {mse}).")

        # See discussion above, unclear how to employ the SKL pipeline for scaling with skorch, manually 
        # scale input before predictions for NN candidate pipelines
        if is_nn_experiment(winner):
            scaler = StandardScaler().fit(X_test)
            X, y = make_nn_set(pd.DataFrame(scaler.transform(X_test)), y_test)
        else: 
            X = X_test
            y = y_test

        preds = winner.predict(X).flatten()
        mse = metrics.mean_squared_error(y, preds, multioutput='raw_values')
        print(f"Test MSE reported : {mse}")
          
        results = build_results(y_test, preds)
        
        print("Best and worst predictions follow: ")
        print(results.head())
        print(results.tail())

        if visualize:           
            heatmap_results(results)          
            show_error_distribution(results)
    else: 
        print(f"No winner reported!")

def main(**args): 
    """
    CLI entry point and arg handler
    """
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group() 
    group.add_argument("-s", "--search", action=argparse.BooleanOptionalAction)
    group.add_argument("-e", "--evaluate", action=argparse.BooleanOptionalAction)
    parser.add_argument("-k", "--splits")
    parser.add_argument("-t", "--threshold")
    parser.add_argument("--conventional", action=argparse.BooleanOptionalAction)
    parser.add_argument("--statcast", action=argparse.BooleanOptionalAction)
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction)
    
    parser.set_defaults(visualize=False)
    parser.set_defaults(threshold=False)
    args = parser.parse_args()
    
    X_train, y_train, X_test, y_test = build_train_test_set(args.conventional, args.statcast) 

    # Search OR evaluate ... nothing really precluding these from being running in series, but
    # there's not really a use-case for it as we are either searching for new configurations or 
    # testing the previously identified (hopefully optimal) ones
    if args.search and (args.visualize is False): 
        search(X_train, y_train, threshold=float(args.threshold), splits=int(args.splits))        
    elif args.evaluate and (args.threshold is False): 
        evaluate(X_train, y_train, X_test, y_test, int(args.splits), args.visualize)
    else: 
        parser.print_help()

if __name__ == "__main__": 
    main()