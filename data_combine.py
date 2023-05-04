import os
import pickle

dir_path = "C:/Users/panxi/PycharmProjects/NeuroScience/650/data/"

def combine_pkl(path):
    res = []
    for file in os.listdir(dir_path):
        # check only text files
        if file.endswith('.pkl'):
            res.append(file)
    data_list = []
    for r in res:
        with open(os.path.join(dir_path,r),'rb') as f:
            data = pickle.load(f)
            data_list = data_list + data
            f.close()
    return data_list