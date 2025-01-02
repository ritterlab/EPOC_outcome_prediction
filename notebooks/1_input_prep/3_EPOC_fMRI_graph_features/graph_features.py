import numpy as np
import pandas as pd
from scipy import stats
import os
import shutil
import glob
import csv

from weighted_graph_functions import *


def get_graph_features(subj, path):
    rs_mat = pd.read_csv(path, sep="\t", header=None)

    rs_mat = rs_mat.dropna(axis = 0, how = 'all')
    rs_mat = rs_mat.dropna(axis = 1, how = 'all')
    rs_mat_np = rs_mat.to_numpy()
    matrix = np.abs(rs_mat_np)
    matrix = remove_self_connections(matrix)

    features = get_weighted_graph_statistics(matrix)
    features['subject_ID'] = subj
    dict_features = [features]

    with open('/home/marijatochadse/2_scripts/EPOC_graphs/{}_rs_graph_features.csv'.format(subj), 'w') as csvfile: 
        writer = csv.DictWriter(csvfile, fieldnames = ['weighted_transitivity',
                                                        'weighted_global_efficiency',
                                                        'weighted_clustering_coefficient_zhang',
                                                        'weighted_clustering_coefficient',
                                                        'weighted_triangle_number',
                                                        'weighted_density',
                                                        'weighted_sw_sigma',
                                                        'weighted_sw_omega',
                                                        'weighted_sw_omega_2',
                                                        'subject_ID']) 
        writer.writeheader() 
        writer.writerows(dict_features) 

    #return(features)

























