# Structure

```
├── README.md                                     # Project documentation and overview
├── data/        
│   └── results/                                  # Contains all ML experiment results
├── notebooks/        
│   ├── 0_explorations/                           
│   │   └── EPOC_overview.ipynb                   # Overview of EPOC dataset
│   ├── 1_input_prep/                             
│   │   ├── EPOC_create_input_files/              
│   │   │   ├── 1_make_demographic*.ipynb         # Create demographic data files
│   │   │   ├── 2_make_sMRI*.ipynb                # Create structural MRI data files
│   │   │   ├── 3_make_fMRI*.ipynb                # Create functional MRI data files
│   │   │   ├── 4_make_graph*.ipynb               # Create graph features fMRI data files
│   │   ├── EPOC_fMRI_graph_features/             
│   │   │   ├── graph_features.py                 # Core graph feature calculations
│   │   │   └── weighted_graph*.py                # Weighted graph analysis functions
│   │   ├── freesurfer_scripts/                   
│   │   └── halfpipe/                             
│   └── 2_run_model/
│       └── run_epoc_ML.ipynb                     # ML Model execution
│   └── 3_results/                                
│       └── p_values_plotting/                    # Statistical analysis and plotting
└── src/        
    ├── MLpipeline/                               
    │   ├── MLpipeline.py                         # Main pipeline class
    │   ├── confounds.py                          # Confound handling utilities
    │   └── runMLpipelines.py                     # Pipeline execution code
    ├── configs.py                                # Global configuration settings
    └── pipeline_configs/                         
        ├── classification_MRI_*.py               # Config files for different experiments
```

- If you see a number prefix on a file name, it is a hint for an execution order. 
