import os, sys, argparse, importlib
from os.path import join
from glob import glob
from datetime import datetime
from copy import deepcopy
from joblib import Parallel, delayed, dump

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# Local imports
from MLpipeline_Marija import *
from confounds import *

def main():
    # load the config file
    parser = argparse.ArgumentParser('runMLpipelines Setting')
    parser.add_argument('config', type=str, help='please add the configuration name')
    parser.add_argument('-d', '--debug', action='store_true', help='runs in debug mode')
    option = parser.parse_args()
    name = option.config
    runDEBUG = option.debug
    pkg = importlib.import_module(f'config.{name}')
    cfg = pkg.Config
    cfg_path = os.path.abspath(sys.modules[cfg.__module__].__file__)
    # set the output directory
    OUTPUT_DIR = cfg_path.replace('config','results').replace('.py','')
    
    # RNG for repeated runs
    rng = np.random.default_rng(seed=cfg.SEED) #edit Marija
    seeds = rng.integers(0, 1000, cfg.N_REPEATS) #edit Marija  #[341, 523, 769]
    
    # DEBUG mode in runMLpipelines if requested
    if runDEBUG:
        print(f'{"="*40}\nRunning DEBUG MODE\n N_OUTER_CV: {cfg.N_INNER_CV} -> 2 \
        \n N_INNER_CV: {cfg.N_OUTER_CV} -> 2 \n N_PERMUTATIONS: {cfg.N_PERMUTATIONS} -> 2 \
        \n N_JOBS: {cfg.N_JOBS} -> 1\n PARALLELIZE: {cfg.PARALLELIZE} -> False \
        \n[Option]\n If use it then also Debug mode both in CounterBalancing and Confound Regression \
        \n{"="*40}')
        cfg.N_INNER_CV = 2
        cfg.N_OUTER_CV = 2
        if cfg.N_PERMUTATIONS > 2:
            cfg.N_PERMUTATIONS = 2
        cfg.N_JOBS = 1 
        cfg.PARALLELIZE = False
    
    # The total number of permutations that are run per trial
    N_PERMUTES_PER_TRIAL = cfg.N_PERMUTATIONS//cfg.N_OUTER_CV

    with Parallel(n_jobs=cfg.N_JOBS) as parallel:
        for h5_file in cfg.H5_FILES:
            start_time = datetime.now()
            h5_name = os.path.basename(h5_file).replace(".h5","")
            # Create the folder in which to save the results
            if runDEBUG: 
                os.system(f'rm -rf {OUTPUT_DIR}/debug_run 2> /dev/null')
                SAVE_DIR = f'{OUTPUT_DIR}/debug_run/{start_time.strftime("%Y%m%d-%H%M")}'
                if not os.path.isdir(SAVE_DIR): os.makedirs(SAVE_DIR)
            else:
                SAVE_DIR = f'{OUTPUT_DIR}/{h5_name}/{start_time.strftime("%Y%m%d-%H%M")}'
                if not os.path.isdir(SAVE_DIR): os.makedirs(SAVE_DIR)
                # Save the configuration file in the folder
                cpy_cfg = f'{SAVE_DIR}/{name}_{start_time.strftime("%Y%m%d_%H%M")}.py'
                os.system(f"cp {cfg_path} {cpy_cfg}")
            
            # read the attributes in the data.h5 file to create all [input -> output] pairs
            with h5py.File(h5_file, 'r') as data:
                # determine the input-output combinations to run from the h5 file
                
                settings = [] #edit Marija
                for nreps in range(cfg.N_REPEATS): #edit Marija, run pipeline n times for better reliability due to small sample size
                    # generate all setting combinations of (1) CONF_CTRL_TECHS, (2) INPUT_OUTPUT combination,
                    # (3) MODEL, and (4) N_OUTER_CV trials so that they can be run in parallel
                    # settings = [] #edited out Marija (inserted above for loop)
                    for key, cfg_i in cfg.ANALYSIS.items():
                        lbl = cfg_i["LABEL"]
                        conf_names = cfg_i["CONFS"]
                        label_names = data.attrs["labels"].tolist()
                        labels = pd.DataFrame({label :np.array(data[label]) for label in label_names})
                        # assert len(label_names)==1, "multiple labels are not supported\
                        # in imagen_ml repository since the commit 7f5b67e95d605f3218d96199c07e914589a9a581."
                        y = lbl if (lbl in label_names) else label_names[0]
                        # prepare the "io"
                        io_combinations = [("X", y)]

                        if cfg_i["RUN_CONFS"]:
                            # skip confound-based analysis if not explicitly requested
                            io_combinations.extend([(c , y) for c in conf_names]) # Same analysis approach
                            io_combinations.extend([("X", c) for c in conf_names]) # Same analysis approach         

                        for conf_ctrl_tech in cfg_i["CONF_CTRL_TECHS"]:
                            for io in io_combinations:
                                for model_pipegrid in cfg_i["MODEL_PIPEGRIDS"]: # pipe=model_pipeline, grid=hyperparameters
                                    # pre-generate the test indicies for the outer CV as they need to run in parallel
                                    if conf_ctrl_tech == "loso":
                                        splitter = LeaveOneGroupOut()
                                        assert splitter.get_n_splits(groups=data['site']) in [7,8]
                                        test_idxs = [test_idx for _,test_idx in splitter.split(data["X"], groups=data['site'])]
                                    else:
                                        splitter = StratifiedKFold(n_splits=cfg.N_OUTER_CV, shuffle=True,
                                                                   random_state=seeds[nreps]) #edit Marija, different seed for each of the n runs
                                        test_idxs = [test_idx for _,test_idx in splitter.split(data["X"], y=labels[y])]
                                        # dd: not performing stratify_by_conf='group' cuz stratification compromises the testset purity as the labels of the testset affects the data splitting and reduces variance in data      
                                    conf_run_names = conf_names 
                                    if conf_ctrl_tech=='baseline-cb': conf_run_names = ['sex', 'site'] # if 'baseline-cb' then control only for 'sex' and 'site' not for any additional variable/s given                           

                                    for trial in range(cfg.N_OUTER_CV):
                                        settings.extend([{"label_name":y, "conf_ctrl_tech":conf_ctrl_tech, 
                                                          "confs": conf_run_names, "io":io, "nickname": key,
                                                          "model_pipegrid":model_pipegrid, "trial":trial, 
                                                          "scoring_list": cfg_i["METRICS"], "task_type": cfg_i["TASK_TYPE"],
                                                          "test_idx":test_idxs[trial], "run_reps": nreps}]) #edit Marija, to avoid overwriting csv files due to identical name
            
            # print the different analysis settings that were preprared above
            print(f'{"="*40}\ntime:  {start_time}\nRunning MLpipeline on file\n ("{h5_file}")\
            \n with {len(settings)} different combinations: \
            \n{" ( #)":2s} {"confound_control-confs":25s}{"input-output":35s} {"ML model":20s} {"out_cv_trial"}')
            for i, setting in enumerate(settings):
                # confs_str = ' '.join(s for s in 
                # confs_list = (str(setting["conf_ctrl_tech"]), str(setting['confs']))
                print(f' ({i:2d}) {str(setting["conf_ctrl_tech"])+": "+",".join(setting["confs"]):25s}{str(setting["io"]):20s}\
                {setting["model_pipegrid"][0].steps[-1][0].replace("model_", ""):10s} \
                cv_i={setting["trial"]:1d}')
            
            # random seeds for inner cv
            rng_inner = np.random.default_rng(seed=cfg.SEED+50) #edit Marija
            repeats_inner = len(settings) #edit Marija
            seeds_inner = rng_inner.integers(0, 1000, repeats_inner) #edit Marija
    

            # runs the experiments with each parameter combination in parallel and save the results in `run_y_i.csv`
            parallel(delayed(
                        conf_corr_run)(
                                    cfg=cfg, h5_file=h5_file, **setting,
                                    save_dir=SAVE_DIR, n_inner_cv=cfg.N_INNER_CV,
                                    parallelize=cfg.PARALLELIZE, n_permutes_per_trial=N_PERMUTES_PER_TRIAL,
                                    permute_only_xy=cfg.PERMUTE_ONLY_XY, 
                                    save_models=cfg.SAVE_MODELS, debug=runDEBUG, random_state=seeds_inner[rnd]) 
                     for rnd, setting in enumerate(settings)) #edit Marija

            # stitch together the csv results that were generated in parallel and save in a single csv file        
            df = pd.concat([pd.read_csv(csv) for csv in glob(SAVE_DIR+"/run_*.csv")], ignore_index=True)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')] # drop unnamed columns            
            df = df.sort_values(["task_type", "inp", "out", "technique", "model", "trial"]) # sort
            df.to_csv(join(SAVE_DIR, "run.csv"), index=False)

            # delete the temp csv files generated in parallel
            os.system(f'rm {SAVE_DIR}/run_*.csv')                                      
            
            runtime=str(datetime.now()-start_time).split(".")[0]
            print(f'TOTAL RUNTIME: {runtime} secs')

##############################################################################################################
            
def conf_corr_run(cfg, h5_file, task_type, scoring_list,
                  conf_ctrl_tech, io, nickname, model_pipegrid, trial, test_idx,
                  label_name, run_reps, save_dir, confs, n_inner_cv, 
                  parallelize, n_permutes_per_trial, permute_only_xy,
                  save_models, debug, random_state=None):
    
    start_time_this_thread = datetime.now()
    conf_ctrl_tech = conf_ctrl_tech.lower()
    i, o = io
    pipe, grid = deepcopy(model_pipegrid)
    model = pipe.steps[-1][0].replace("model_", "")
    analysis_name = nickname + "_" + o + "_" + model
    print(f'{"-"*38}\nStarting a new pipeline with setting:\
    \n conf_ctrl_tech={conf_ctrl_tech}, io={io}, model={model}, outer_cv_trial={trial}')
    
    m = MLpipeline(parallelize, random_state=random_state)
    # load X, y and confounds
    m.load_data(h5_file, y=label_name, confs=confs, group_confs=True)
    
    # randomly split data into training and test set
    m.train_test_split(test_idx=test_idx)
    
    # fill in missing values edit Marija
    if cfg.FILLED_MISSING == True:
        print('Fill missing set to true')
        m.fill_missing_vals()
        
    # PCA input data edit Marija
    if cfg.PCA == True:
        print('Running PCA on input data')
        m.PCA_data(n_components = cfg.N_COMPONENTS)
                                        
    ### <START> Special conditions for each confound correction conf_ctrl_tech
    conf_corr_params = {}  
    stratify_by_conf = None
    n_samples_cc = m.n_samples_tv
    
    # 1) CounterBalancing
    if "cb" in conf_ctrl_tech:
        # Counterbalance for both sex and site, which is "group"
        oversample = True
        if conf_ctrl_tech == "under-cb":
            oversample=False
        elif conf_ctrl_tech == "overunder-cb":
            oversample=None
        else:
            oversample=True
        cb = CounterBalance(oversample, random_state=random_state, debug=debug)
        pipe.steps.insert(-1, ("conf_corr_cb", cb))
        conf_corr_params.update({"conf_corr_cb__groups": m.confs["group"]})
        # when output is not the label 'y', still perform counterbalancing across the label 'y'
        if (o in confs): conf_corr_params.update({"conf_corr_cb__cb_by": m.y}) 
        # calculate info about how CB changes the training sample size
        n_samples_cc = len(cb._get_cb_sampled_idxs(groups=m.confs["group"], cb_by=m.y)) 
        
    # 2) Confound Regression Categorical
    elif (conf_ctrl_tech in ["cr"]) and (i == "X"):
        cr = ConfoundRegressorCategoricalX(debug=debug)
        pipe.steps.insert(-1, ("conf_corr_cr", cr))
        conf_corr_params.update({"conf_corr_cr__groups": m.confs["group"]})
    
    # 3) Confound Regression Continous 
    elif "cc" in conf_ctrl_tech:
        cc = ConfoundRegressorContinuousX(debug=debug)
        pipe.steps.insert(-1, ("conf_corr_crc", cc))
        conf_corr_params.update({"conf_corr_crc__age": m.confs["age"]})
    
    ### <END> Special conditions for each conf_ctrl_conf_ctrl_tech
    if (i in confs): m.change_input_to_conf(i, onehot=True)
    if (o in confs): m.change_output_to_conf(o)
    
    # run permutation for other than X-y experiments?
    if permute_only_xy and ((i in confs) or (o in confs)): n_permutes_per_trial = 0
        
    # run the actual classification pipeline including the hyperparameter tuning
    run = m.run(pipe, grid, task_type, scoring_list, 
                 n_splits=n_inner_cv, 
                 conf_corr_params=conf_corr_params,
                 stratify_by_conf=stratify_by_conf,
                 permute=n_permutes_per_trial)
    
    # prepare results
    result = {
        "task_type" : task_type,
        "model" : model,
        "analysis" : analysis_name,
        "repetition" : run_reps,
        "trial" : trial,
        "inp" : i,
        "out" : o,
        "technique" : conf_ctrl_tech,
        "confs" : confs,
        "n_samples":(m.n_samples_tv + m.n_samples_test),
        "n_samples_cc":(n_samples_cc + m.n_samples_test),
    }
    
    # append results
    result.update(run)
    runtime = int((datetime.now() - start_time_this_thread).total_seconds())
    result.update({"runtime":runtime})
    
    df = pd.DataFrame([result])
    df.to_csv(join(save_dir, f'run_{label_name}_{random_state}_{run_reps}_{trial}.csv')) #edit Marija
    
    # save models only for X-y experiments with conf_ctrl_tech == CounterBalance
    if save_models and (i not in confs) and (o not in confs) and (conf_ctrl_tech!="baseline"):
        dump(m.estimator, join(save_dir, f'{model}_{conf_ctrl_tech}_{trial}.model'))
    
    val_score = result['val_metric']*100
    total_runtime = str(datetime.now() - start_time_this_thread).split(".")[0]
    print(f'Finished after {total_runtime}s with val_score = {val_score:.2f}')
    
if __name__ == "__main__": main()