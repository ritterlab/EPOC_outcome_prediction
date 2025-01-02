from os.path import join
import h5py
import numpy as np
import pandas as pd
import statistics as st
from datetime import datetime 
from joblib import Parallel, delayed

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import get_scorer, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import PCA


class MLpipeline:
    
    def __init__(self, parallelize=True, random_state=None):
        """ Define parameters for the MLLearn object.
        Args:
            "h5file" (str): Full path to the HDF5 file.
            "conf_list" (list of str): the list of confound names to load from the hdf5.
            random_state (int): A random state to get reproducible results
        """        
        self.random_state = random_state
        self.parallelize = parallelize
        self.confs = {}  
        
        self.X_test = None
        self.y_test = None
        self.confs_test = {} 
        self.sub_ids = None
        self.sub_ids_test = None
        
        # These are defined to calculate later.
        self.func_auc = get_scorer("roc_auc")
        self.func_r2 = get_scorer('r2')
        

    def load_data(self, f, X="X", y="y", confs=[], group_confs=False):
        """ Load the data from the HDF5 file, 
            The neuroimaging data/ input data is saved under "X", the label under "y" and 
            the confounds from the dict self.confs
            If group_confs==True, groups all confounds into one numpy array and saves in
            self.confs["group"]
        """
        if isinstance(f, dict):
            df = f
        else:
            df = h5py.File(f, "r")
        self.X = np.array(df[X])
        self.y = np.array(df[y])
        self.sub_ids = np.array(df["i"])
        if df.get('scale_info'): #edit Marija
            self.scale_info = list(df["scale_info"]) #edit Marija
        if df.get('confs_scale_info'): #edit Marija
            self.confs_scale_info = list(df["confs_scale_info"]) #edit Marija
        
        # self.X needs to be flattened as sklearn expects 2D input.
        if self.X.ndim > 2:
            self.X = self.X.reshape([self.X.shape[0], np.prod(self.X[0].shape)])
        elif self.X.ndim == 1:
            self.X = self.X.reshape(-1,1) # sklearn expects 2D input
            
        # Load the confounding variables into a dictionary, if there are any
        for c in confs:
            if c in df.keys():
                v = np.array(df[c])
                self.confs[c] = v 
                
                if group_confs:
                    if "group" not in self.confs:
                        self.confs["group"] = v
                    else:
                        self.confs["group"] = v + 100*self.confs["group"]
            else:
                print(f"[WARN]: confound {c} not found in hdf5 file.")          
        

    def train_test_split(self, test_idx=[], test_size=0.25, stratify_by_conf=None):
        """ Split the data into a test set (_test) and a train+val set.

        Args:
            stratify_group (str, optional): A string that indicates a confound name. 
                Get confounds list available from the dict self.confs.
                If a confound is selected, subjects will be stratified according 
                to confound and outcome label. Defaults to None, in which case 
                subjects are only stratified according to the outcome label.
        """
        # if test_idx are already provided then dont generate the test_idx yourself
        if not len(test_idx): 
            # by default, always stratify by label
            stratify = self.y
            if stratify_by_conf is not None:
                stratify = self.y + 100*self.confs[stratify_by_conf]

            _, test_idx = train_test_split(range(len(self.X)), 
                                                      test_size=test_size, 
                                                      stratify=stratify, 
                                                      shuffle=True, 
                                                      random_state=self.random_state)            
        test_mask = np.zeros(len(self.X), dtype=bool)
        test_mask[test_idx] = True
            
        self.X_test = self.X[test_mask]
        self.y_test = self.y[test_mask]
        self.sub_ids_test = self.sub_ids[test_mask]
        self.X = self.X[~test_mask]
        self.y = self.y[~test_mask]
        self.sub_ids = self.sub_ids[~test_mask]
        
        for c in self.confs:
            self.confs_test[c] = self.confs[c][test_mask]
        for c in self.confs:
            self.confs[c] = self.confs[c][~test_mask]
        
        self.n_samples_tv = len(self.y)
        self.n_samples_test = len(self.y_test)
        
   
    # fill missing values edit Marija
    def fill_missing_vals(self):
        """ Fill in missing values in outer CV loop."""
        
        for X_col in range(self.X.shape[1]):            
            if np.isnan(self.X[:,X_col]).any(axis=0):
                if self.scale_info[X_col] == b'categorical':
                    fill_val = st.mode(self.X[:,X_col][~np.isnan(self.X[:,X_col])])
                    self.X[:,X_col][np.isnan(self.X[:,X_col])] = fill_val
                elif self.scale_info[X_col] == b'numerical':
                    fill_val = st.median(self.X[:,X_col][~np.isnan(self.X[:,X_col])])
                    #fill_val = np.nanmean(self.X[:,X_col])
                    self.X[:,X_col][np.isnan(self.X[:,X_col])] = fill_val
                
            if np.isnan(self.X_test[:,X_col]).any(axis=0):
                if self.scale_info[X_col] == b'categorical':
                    fill_val = st.mode(self.X[:,X_col][~np.isnan(self.X[:,X_col])])
                    self.X_test[:,X_col][np.isnan(self.X_test[:,X_col])] = fill_val
                elif self.scale_info[X_col] == b'numerical':
                    fill_val = st.median(self.X[:,X_col][~np.isnan(self.X[:,X_col])])
                    #fill_val = np.nanmean(self.X[:,X_col])
                    self.X_test[:,X_col][np.isnan(self.X_test[:,X_col])] = fill_val
                                           
        if self.confs:
            for X_col, c in enumerate(self.confs):
                if np.isnan(self.confs[c]).any(axis=0):
                    if self.confs_scale_info[X_col] == b'categorical':
                        fill_val = st.mode(self.confs[c][~np.isnan(self.confs[c])])
                        self.confs[c][np.isnan(self.confs[c])] = fill_val
                    elif self.confs_scale_info[X_col] == b'numerical':
                        fill_val = st.median(self.confs[c][~np.isnan(self.confs[c])])
                        self.confs[c][np.isnan(self.confs[c])] = fill_val

                if np.isnan(self.confs_test[c]).any(axis=0):
                    if self.confs_scale_info[X_col] == b'categorical':
                        fill_val = st.mode(self.confs[c][~np.isnan(self.confs[c])])
                        self.confs_test[c][np.isnan(self.confs_test[c])] = fill_val
                    elif self.confs_scale_info[X_col] == b'numerical':
                        fill_val = st.median(self.confs[c][~np.isnan(self.confs[c])])
                        self.confs_test[c][np.isnan(self.confs_test[c])] = fill_val
        
    def PCA_data(self, n_components):
        
        #X_train_scaled = (self.X - self.X.mean(axis=0)) / self.X.std(axis=0)
        #X_test_scaled = (self.X_test - self.X_test.mean(axis=0)) / self.X_test.std(axis=0)
        
        X_train_scaled = StandardScaler().fit_transform(self.X)
        X_test_scaled = StandardScaler().fit_transform(self.X_test)
        
        pca = PCA(n_components=n_components)
        #pca_test = PCA(n_components=n_components)
        
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        self.X = X_train_pca
        self.X_test = X_test_pca

        
    def transform_data(self, func):
        '''performs func.transform() (see sklearn structures)
                on either (a) all the data loaded, if performed before train_test_split()
                    or on (b) the train+val subset, if performed after train_test_split()'''
        # first check if train_test_split has already been performed
        if self.X is not None:
            self.X = func.transform(self.X)
            self.y = func.transform(self.y)

            for c in self.confs:
                self.confs[c] = func.transform(self.confs[c])
           
        
    def print_data_size(self):
        if self.X is None:
            print("data is not loaded yet.. run MLpipeline.load_data()")
        else:
            print(f"Trainval data info:\t X ={self.X.shape} \t y ={self.y.shape} \t confs={self.confs.keys()}")
        if self.X_test is None:
            print("test data is not loaded yet.. run MLpipeline.load_data() and then MLpipeline.train_test_split()")
        else:
            print(f"    Test data info:\t X ={self.X_test.shape} \t y ={self.y_test.shape} \t n(trainval)={self.n_samples_tv}, n(test)={self.n_samples_test}")
            
        
    def change_input_to_conf(self, c, onehot=True):
        """ Change the inputs of the model to a vector containing a confound.

        Args:
            c (str): The name of the confound from the list loaded in conf_dict. 
                            Get confounds list available from the dict self.confs.
            onehot (bool, optional): Whether to one-hot encode the new input. This should be 
                enabled for linear/logistic regression models. Defaults to True.
        """
        assert self.y_test is not None, "self.train_test_split() should be run before calling self.change_input_to()"
        self.X = self.confs[c].reshape(-1, 1)
        self.X_test = self.confs_test[c].reshape(-1, 1)
        
        if onehot:
            self.X = OneHotEncoder(sparse=False).fit_transform(self.X)
            self.X_test = OneHotEncoder(sparse=False).fit_transform(self.X_test)


    def change_output_to_conf(self, c):
        """ Change the targets (outputs) of the classifier to a confound vector.

        Args:
            c (str): The name of the confound which is loaded in self.confs, 
                Get confounds list using self.confs().
        """        
        assert self.y_test is not None, "self.train_test_split() should be run before calling self.change_output_to()"
        self.y = self.confs[c]
        self.y_test = self.confs_test[c]
        
            
    @ignore_warnings(category=ConvergenceWarning)
    def run(self, pipe, grid, task_type='classification', scoring_list=None,
            n_splits=5, stratify_by_conf=None, conf_corr_params={}, 
            permute=0, verbose=2):
        """ The main function to run the classification. 

        Args:
            pipe (sklearn Pipeline): a pipeline containing preprocessing steps 
                and the classification model. 
            grid (dict): A dict of hyperparameter lists that will be passed to 
                sklearn's GridSearchCV() function as 'param_grid'
            permute (int, optional): Number of permutations to perform. Defaults to 0 i.e.
                no permutations are performed.

        Returns:
            results (dict): A dictionary containing classification metrics for the 
                best parameters.
            best_params (dict): The best parameters found by grid search.
        """        
        assert self.X_test is not None, "self.train_test_split() has not been run yet. \
First split the data into train and test data"
        
        # create the inner cv folds on the trainval split
        cv = StratifiedKFold(n_splits, shuffle=True, random_state=self.random_state)
        # stratify the inner cv folds by the label and confound groups
        stratify = self.y
        if stratify_by_conf is not None:
            stratify = self.y + 100*self.confs[stratify_by_conf]            
        cv_splits = cv.split(self.X, stratify)
        
        n_jobs = n_splits if (self.parallelize) else None 
        # grid search for hyperparameter tuning with the inner cv         
        gs = GridSearchCV(estimator=pipe, param_grid=grid, n_jobs=n_jobs,
                          cv=cv_splits, scoring=scoring_list,
                          return_train_score=True, refit=True, verbose=verbose)
        # fit the estimator on data
        gs = gs.fit(self.X, self.y, **conf_corr_params)
        
        self.estimator = gs.best_estimator_
        
        # store scores
        train_score = np.mean(gs.cv_results_["mean_train_score"])
        inner_cv_score = gs.best_score_ # mean cross-validated score of the best_estimator
        val_score = gs.score(self.X_test, self.y_test)
        val_lbls = self.y_test
        val_ids = self.sub_ids_test
        train_ids = self.sub_ids #edit Marija
        
        if "classification" in task_type:
            # save the predicted probability scores on the test subjects (for trying other metrics later)
            val_probs = np.around(gs.predict_proba(self.X_test), decimals=4)
            # Calculate AUC if label is binary
            roc_auc = np.nan        
            if len(np.unique(self.y_test))==2:
                roc_auc = self.func_auc(self.estimator, self.X_test, self.y_test)   
                
        elif "regression" in task_type:
            # save the predicted probability scores on the test subjects (for trying other metrics later)
            val_probs = np.around(gs.predict(self.X_test), decimals=4)
            # Calculate R^2 score if label is continuous
            r2_score = self.func_r2(self.estimator, self.X_test, self.y_test)
                
        # if permutation is requested, then calculate the test statistic after permuting y  
        # TODO: permutation on regression
        results_pt = {}        
        if permute:
            with Parallel(n_jobs=n_jobs) as parallel:
                # run parallel jobs on all cores at once
                pt_scores = parallel(delayed(
                                MLpipeline._one_permutation)(
                                    self.X, self.y, self.X_test, self.y_test,
                                    pipe, grid, fit_params=conf_corr_params, confs_test=self.confs_test,
                                    n_splits=n_splits, score_func=scoring_list, score_func_auc=self.func_auc,
#                                     permute_y=False, permute_x=True, permute_y_test=False, permute_x_test=False         
                )  for _ in range(permute))
            
            pt_scores  = np.array(pt_scores)           
            results_pt = {"permuted_test_score" : pt_scores[:,0].tolist()}
            
            if not np.isnan(pt_scores[:,1]).all():
                results_pt.update({"permuted_roc_auc": pt_scores[:,1].tolist()})
            
        results = {
            "model_metric" : scoring_list,
            "innerval_metric" : inner_cv_score, 
            "m__params" : gs.best_params_,
            "train_metric" : train_score, 
            "val_metric" : val_score,
            "val_ids" : val_ids.tolist(),
            "train_ids": train_ids.tolist(),
            "val_lbls" : val_lbls.tolist(),
            "val_preds" : val_probs.tolist(),
            **results_pt,
            }
        
        if "classification" in task_type:
            classification = {
                "other_metrics" : dict(roc_auc = roc_auc)
            }
            results.update(classification)
        
        elif "regression" in task_type:
            regression = {
                "other_metrics" : dict(r2_score = r2_score)
            }
            results.update(regression)
        return results
    

    @staticmethod
    def sensitivity_score(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tp/(tp+fn)

    
    @staticmethod
    def specificity_score(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn/(tn+fp) 
    
    
    @staticmethod
    def _shuffle(arr, groups=None):
        """ 
        Permute a vector or array y along axis 0.
        Args:
            arr (numpy array): The to-be-permuted array. The array will be permuted
                along axis 0.  
            groups (str): If set, the permutations of y will only happen within members
                of the same group. Get confounds list available from the dict self.confs
        Returns:
            y_permuted: The permuted array y_permuted. """    
        if groups is None:
            indices = np.random.permutation(len(arr))
        else:
            indices = np.arange(len(groups))
            for group in np.unique(groups):
                this_mask = (groups == group)
                indices[this_mask] = np.random.permutation(indices[this_mask])
        return arr[indices]
    
    
    @staticmethod
    def _one_permutation(X, y, X_test, y_test, 
                        estimator, grid, fit_params={}, confs_test=None,
                        n_splits=5, score_func=None, score_func_auc=get_scorer("roc_auc"),
                        permute_test=False):

        """ Run the standard gridsearch+cv pipeline once after permuting X or y
        and evaluate on the test set X_test, y_test
        Args:

        Returns:
            pt_score (float): Balanced accuracy score from permuted samples.
            pt_score_auc (float): ROC AUC score from permuted samples.
            d2_* (float): D2 scores for fitting the PBCC models with different
                independent variables *. See function pbcc().
                
        TODO: permutation on regression
        """
        # By default X is shuffled rather than the labels y as the relationship 
        # between confound and label should be maintained for PBCC. refer Dinga et al., 2020.
        X = MLpipeline._shuffle(X)
            
        if permute_test: 
            X_test = MLpipeline._shuffle(X_test) 
            
        # create the inner crossvalidation folds on the trainval split
        # random state is not fixed because each permutation must be completely random
        cv = StratifiedKFold(n_splits, shuffle=True, random_state=None) 
        # grid search for hyperparameter tuning with the inner cv 
        gs = GridSearchCV(estimator, param_grid=grid, n_jobs=None,
                          cv=cv, scoring=score_func,
                          return_train_score=False, refit=True, verbose=0)
        
        # disable the random_state in the confound correction to get varying scores
        if "conf_corr_cb__cb_by" in fit_params:
            estimator["conf_corr_cb"].random_state = None
            
        # fit the estimator on data            
        gs = gs.fit(X, y, **fit_params)

        pt_score = gs.score(X_test, y_test)
        
        # calc auc score
        pt_score_auc = np.nan
        if len(np.unique(y)) == 2:
            pt_score_auc = score_func_auc(gs.best_estimator_, X_test, y_test)
        return [pt_score, pt_score_auc]