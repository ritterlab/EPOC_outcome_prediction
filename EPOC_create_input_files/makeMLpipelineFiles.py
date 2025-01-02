import numpy as np
import pandas as pd
import pyreadstat
import seaborn as sns
from scipy import stats
import h5py



class makeMLpipelineFiles:
    
    def __init__(self, file_path):
        """
        file_path: file path to .csv file containing data from all input variables, confounding variables, 
        as well as the outcome variable in original form,
        can contain missing values.
        
        Only requirement: the .csv file should only contain subjects to be included in the analysis
        """
    
        self.file_path = file_path
        
        
    def format_data(self, y, variables, confs=[]):
        """
        IN:
        y: outcome variable name, input format as string
        variables: input variable names to be used for prediction, excluding confounding variables, input format as list
        confs: confounding variable names, input format as list
        
        OUT:
        self.df: dataframe including input variables, excluding confounding variables and outcome variable, 
                 all values are floats, categorical variables coded numerically
        self.df_confs: dataframe including confounding variables,
                       all values are floats, categorical variables coded numerically
        self.idx: list with subjetc indices, to be used for train test splitting
        self.scale_info: list with scale of measurement info for input variables, excluding confounding variables
        self.confs_scale_info: list with scale of measurement info for confounding variables
        """
    
        df_original = pd.read_csv(self.file_path, sep=",")
        df_temp_y = df_original.copy()
        df_temp = df_temp_y.drop(columns=y)
        
        scale_info = []
        confs_scale_info = []   
        
        var_names = df_temp.columns.values.tolist()
        
        for var_i in var_names:
            if df_temp.dtypes[var_i] == 'float64':
                if var_i in variables:
                    scale_info.append('numerical')
                elif var_i in confs:
                    confs_scale_info.append('numerical')

            elif df_temp.dtypes[var_i] == 'object':
                var_vals = list(df_temp[var_i].unique()) #get unique values that this variable can take on
                var_n = len(df_temp[var_i].unique()) #generate integers for unique values
                var_idx = list(np.arange(var_n))
                df_temp[var_i].replace(var_vals, var_idx, inplace=True) #replace with integers
                if var_i in variables:
                    scale_info.append('categorical')
                elif var_i in confs:
                    confs_scale_info.append('categorical')
        
        if df_temp_y.dtypes[y] == 'object':
            yvals = list(df_temp_y[y].unique()) #get unique values that this variable can take on
            yvar_n = len(df_temp_y[y].unique()) #generate integers for unique values
            yvar_idx = list(np.arange(yvar_n))
            df_temp_y[y].replace(yvals, yvar_idx, inplace=True) #replace with integers
                     
        
        df = df_temp[variables].copy()
        df_confs = df_temp[confs].copy()
        df_y = df_temp_y[y].copy()
        idx = list(df.index.values) #the index values per subject
        
        self.y = y
        self.variables = variables
        self.confs = confs
        self.df = df
        self.df_confs = df_confs
        self.df_y = df_y
        self.idx = idx
        self.scale_info = scale_info
        self.confs_scale_info = confs_scale_info
              
        print(self.idx)
        #return df.astype('float64'), df_confs.astype('float64'), scale_info, confs_scale_info, idx
        
        
    def create_h5(self, filename):
        """
        IN:
        filename: path and file name of output .h5 file to be generated
        
        OUT:
        file.h5 file
        """
    
        f = h5py.File(filename, "w")

        # input variables
        f.create_dataset('X', data=self.df)
        #f.attrs['X_col_names']= list(self.df.columns) #or self.variables
                
        # output variable
        f.create_dataset(self.y, data=self.df_y)
        f.attrs['labels']= [self.y]
        
        # confounding variables
        if self.confs:
            if 'Alter' in self.confs:
                self.confs = list(map(lambda x: x.replace('Alter', 'age'), self.confs)) 
            for conf in self.confs:
                if conf == 'age':
                    f.create_dataset(conf, data=self.df_confs.loc[self.idx, "Alter"])
                else:
                    f.create_dataset(conf, data=self.df_confs[conf])
            f.attrs['confs']= [self.confs]        
            f.create_dataset('confs_scale_info', data=self.confs_scale_info)
        
        # scale info
        f.create_dataset('scale_info', data=self.scale_info)
        
        # indeces
        f.create_dataset('i', data=self.idx)
        
        
        print(f.keys(), f.attrs.keys(), "confs:", self.confs)
        f.close()

 