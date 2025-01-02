from imblearn.pipeline import Pipeline

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingRegressor

import pickle

# (0) general local variables created for ease-of-use in the Config class
# the model settings that will be common across all label analysis. 
# This is exists to avoid the need to repeat these settings.
Classification_model_settings = [
    #(pipeline, grid) for Logistic Regression classifier
    (
        Pipeline([
            ("varth", VarianceThreshold()),
            ("scale", StandardScaler()),
            ("model_LR(c)", LogisticRegression(max_iter=1000))
        ]),
        {"model_LR(c)__C" : [10000, 1000, 100, 1.0, 0.001, 0.0001]}, 
    ),
     #(pipeline, grid) for linear SVM classifier
    (
        Pipeline([
            ("varth", VarianceThreshold()),
            ("scale", StandardScaler()),
            ("model_SVM-lin", SVC(kernel="linear", max_iter=10000, probability=True))
        ]),
        {"model_SVM-lin__C" : [10000, 1000, 100, 1.0, 0.001, 0.0001]},     
    ),
    # (pipeline, grid) for SVM classifier with rbf kernel
    (
        Pipeline([
            ("varth", VarianceThreshold()),
            ("scale", StandardScaler()),
            ("model_SVM-rbf", SVC(kernel="rbf", probability=True))
        ]),
        {"model_SVM-rbf__C" : [10000, 1000, 100, 1.0, 0.001, 0.0001],
         "model_SVM-rbf__gamma" : ['scale', 'auto']}
    ),
    # (pipeline, grid) for GradientBoosting classifier
    (
        Pipeline([
            ("varth", VarianceThreshold()),
            ("scale", StandardScaler()),
            ("model_GB", XGBClassifier(n_estimators=100, max_depth=5, subsample=1.0, 
                                           use_label_encoder=True,  eval_metric='logloss'))
       ]),
        {"model_GB__learning_rate" : [0.01, 0.05, 0.1, 0.25]} 
    )
]

Regression_model_settings = [
    # (pipeline, grid) for Linear Regression Regressor
    (
        Pipeline([
            ("model_LR(r)", LinearRegression())
        ]),
        {'model_LR(r)__fit_intercept': [True,False]},
    ),
    # (pipeline, grid) for Lasso Regressor
    (
        Pipeline([
            ("model_Lasso", Lasso())
        ]),
        {'model_Lasso__alpha': [0.1]}
    ),
    # (pipeline, grid) for Gradient Bossting Regressor
    (
        Pipeline([
            ("model_GBR", GradientBoostingRegressor())
        ]),
        {'model_GBR__learning_rate': [0.01,0.02,0.03,0.04],
         # 'model_GBR__subsample'    : [0.9, 0.5, 0.2, 0.1],
         # 'model_GBR__n_estimators' : [100,500,1000, 1500],
         # 'model_GBR__max_depth'    : [4,6,8,10]
        },
    ),
]


class Config:
    '''
    Configuration Settings
    ----------------------
    A template Configuration file for runMLpipelines wrapper.
    Please copy and save new {NAME}.py and fill out your own setting below.
    Before write it down setting, please double check the importing library above.
    If you have any question, please feel free to contact JiHoon.
    
    # CONFIG start ################################################################################################    
    
    ## DATASET
    H5_FILES        : The dataset file
                      Here you can load the which HDF5 files you want to include in analysis.
                      You should know which X, labels(s), and confound(s) contained in HDF5 file(s).
                      Please load the HDF5 file(s) with aboslute path as a list.
                      --example--
                      H5_FILES = ['/ritter/share/data/UKBB_2020/h5files/idps-l-highalcl0u2-c-sex-age-n13465.h5']
    
    ## ANALYSIS
    ' '             : NAME
                      Please provide the unique name, analysis configuration without model using "_".
                      --example--
                      'classification_cb_sex'
                      
    : dict(         : ANALYSIS
                      Here you define training settings specific to LABEL for the anlysis.
                      Please provide the analysis configuration
    
    LABEL           : Which columns in the HDF5 file should be used as 'y' / labels in the analysis.
                      If the label is empty or wrong, then automatically select first label in the candidates labels.
                      --Note--
                      '' labely setting would be useful to run differnet label of the HDF5 file(s) at once.
                      --example--
                      LABEL = ''
    
    TASK_TYPE       : Task type.
                      Choose either "classification" or "regression".
                      (TODO): supports "mutli-classification", "clustering" etc.
                      --example--
                      TASK_TYPE = 'classification'
    
    MODEL_PIPEGRIDS : The ML pipelines to run and their corresponding hyperparameter grids as tuples,
                      i.e. (pipeline, grid)

    METRICS         : Evaluating the quality of a model’s predictions.
                      Please provide the metric name.
                      For more detail, please find the link:
                       https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
                      --example--
                      METRICS = 'balanced_accuracy'
    
    ### OPTION
    RUN_CONFS       : If True, run X->y, c->y, X->c.
                      --example--
                      RUN_CONFS = True
                      
    CONF_CTRL_TECHS : Confound control (cc) techniques.
                      Choose from ["baseline", "cb", "baseline-cb", "cr", "loso", "cc"]
                      No cc then, select "baseline".
                      --example--
                      CONF_CTRL_TECHS = ['baseline', 'cb']
                      
    CONFS           : List of the Confound name
                      --example--
                      CONFS = ['sex', 'site']
    )
    
    ## SETTING         
    N_INNER_CV      : Number of folds in inner crossvalidation used for hyperparameter tuning
                      --example--
                      N_INNER_CV = 5
    
    N_OUTER_CV      : Number of folds in inner crossvalidation for test score estimation
                      --example--
                      N_OUTER_CV = 7
    
    N_PERMUTATIONS  : Total number of permutation test to run. Set to 0 to not perform any permutations.
                      --example--
                      N_PERMUTATIONS = 0
    
    PERMUTE_ONLY_XY : If True, permute only xy
                      --example--
                      PERMUTE_ONLY_XY = True
    
    N_JOBS          : Parallel jobs
                      --example--
                      N_JOBS = 2
             
    PARALLELIZE     : If True, within each MLPipeline trial, paralleize the permutation test runs.
                      --example--
                      PARALLELIZE = True
                      
    SAVE_MODELS     : If True, saves the final trained models but only for inp-out=={X-y} and conf_ctrl_tech='CB'
                      --example--
                      SAVE_MODELS = False
                      
    IDCS            : list of test, train and val indices to feed into MLpipeline, or empty list #*

    # CONFIG ends  ################################################################################################
    '''

    H5_FILES = [ 
        #'/ritter/share/data/EPOC/h5_files/EEGneutrneg_OUTRemission_CONFSSex.h5',
        #'/ritter/share/data/EPOC/h5_files/TAB_OUTRemission_CONFSSex.h5',
        #'/home/marijatochadse/1_data/EPOC/h5_files/therapy_OUTRemission_CONFSSex.h5',
        #'/home/marijatochadse/1_data/EPOC/h5_files/comorbid_OUTRemission_CONFSSex.h5',
        #'/home/marijatochadse/1_data/EPOC/h5_files/clin_OCD_OUTRemission_CONFSSex.h5',
        #'/home/marijatochadse/1_data/EPOC/h5_files/clin_other_OUTRemission_CONFSSex.h5',
        #'/ritter/share/data/EPOC/h5_files/YBOCS_T0_OUTRemission_CONFSSex.h5',
        #'/home/marijatochadse/1_data/EPOC/h5_files/clin_ybocs_EEG_OUTRemission_CONFSSex.h5',
        #'/ritter/share/data/EPOC/h5_files/TAB_noclin_OUTRemission_CONFSSex.h5',
        '/ritter/share/data/EPOC/h5_files/graph_fMRI_OUTRemission_CONFSSex.h5',

    ]
    
    
    ANALYSIS = {
        # classification case
        'classification_baseline' : dict(
            LABEL = 'Remission',
            TASK_TYPE='classification',
            METRICS='balanced_accuracy',
            MODEL_PIPEGRIDS = Classification_model_settings,
            RUN_CONFS = True,
            CONF_CTRL_TECHS = ['cb'], #baseline
            CONFS = ['Geschlecht'],
        ),
                
    }
    
    # SETTING
    SEED = 42 #edit Marija
    N_REPEATS = 10 #edit Marija
    N_INNER_CV = 5
    N_OUTER_CV = 7
    N_PERMUTATIONS = 1000
    PERMUTE_ONLY_XY = False
    N_JOBS = 10
    PARALLELIZE = True
    SAVE_MODELS = False
    FILLED_MISSING = True #edit Marija
    PCA = False #edit Marija
    N_COMPONENTS = None # edit Marija