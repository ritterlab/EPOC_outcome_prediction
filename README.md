# Introduction

- What is the project

# Structure, 

```
├── README.md                                     # Project documentation and overview
├── data/        
│   └── results/                                  # Contains all ML experiment results
├── notebooks/        
│   ├── 0_explorations/                           # Initial data exploration notebooks
│   │   └── EPOC_overview.ipynb                   # Overview of EPOC dataset
│   ├── 1_input_prep/                             # Data preparation notebooks
│   │   ├── EPOC_create_input_files/              # Scripts to create input datasets
│   │   │   ├── 1_make_demographic*.ipynb         # Create demographic data files
│   │   │   ├── 2_make_sMRI*.ipynb                # Process structural MRI data
│   │   │   ├── 3_make_fMRI*.ipynb                # Process functional MRI data
│   │   │   ├── 4_make_graph*.ipynb               # Create graph features from fMRI
│   │   ├── EPOC_fMRI_graph_features/             # Graph feature extraction tools
│   │   │   ├── graph_features.py                 # Core graph feature calculations
│   │   │   └── weighted_graph*.py                # Weighted graph analysis functions
│   │   ├── freesurfer_scripts/                   # Scripts for FreeSurfer processing
│   │   │   ├── 1_subj_level*.sh                  # Individual subject processing
│   │   │   ├── 2_loop_subjs*.sh                  # Batch processing scripts
│   │   │   └── 6_BNA_to_table*.sh                # Convert BNA atlas data to tables
│   │   └── halfpipe/                             # fMRI preprocessing pipeline configs
│   ├── 2_modeling/                               # Model development notebooks
│   └── 3_results/                                # Analysis and visualization
│       └── p_values_plotting/                    # Statistical analysis and plotting
└── src/        
    ├── MLpipeline/                               # Core ML pipeline implementation
    │   ├── MLpipeline.py                         # Main pipeline class
    │   ├── confounds.py                          # Confound handling utilities
    │   └── runMLpipelines.py                     # Pipeline execution code
    ├── configs.py                                # Global configuration settings
    └── pipeline_configs/                         # Experiment-specific configurations
        ├── classification_MRI_*.py               # Config files for different experiments
```

- If you see a number prefix on a file name, it is a hint for an execution order. 