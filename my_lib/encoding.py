import numpy as np
import pickle
from math import sin, cos, pi
# from datetime import date
# import os
# import pandas as pd


# # from .constants import *



class DataEncoder:
    
    def fit_transform(self, df):
        
        self.TCODE_TO_NUM = dict([(tc, i) for i, tc in enumerate(df.tcode.unique())])
        self.NUM_TO_TCODE = dict([(i, tc) for i, tc in enumerate(df.tcode.unique())]) 
        df["tcode_num"] = df["tcode"].apply(lambda x: self.TCODE_TO_NUM[x])
        
        
        df["log_amount"] = np.log10(df["amount"]+1)
        self.LOG_AMOUNT_SCALE = df["log_amount"].std()
        df["log_amount_sc"] = df["log_amount"] / self.LOG_AMOUNT_SCALE
        
        self.TD_SCALE = df["td"].std()
        df["td_sc"] = df["td"] / self.TD_SCALE
        
        self.ATTR_SCALE = df["age"].std()
        df["age_sc"] = df["age"] / self.ATTR_SCALE
        
        self.START_DATE = df["datetime"].min()
        
        self.n_tcodes = len(self.TCODE_TO_NUM)
        
        

def preprocess_df(df, ds_suffix = None):
    de = DataEncoder()
    de.fit_transform(df)
    
    if ds_suffix == None:
        print("No ds_suffix set. Using ds_suffix = 'default'. (ds_suffix is used for keeping track of different dataset versions)")
        ds_suffix = 'default'
    
    
    with open(f"stored_data/DataEncoder-{ds_suffix}.pickle", "wb") as f:
        pickle.dump(de, f) 
        print("Wrote encoding info to", f.name)
        
    return de

        

def load_data_encoder(ds_suffix):
    with open(f"stored_data/DataEncoder-{ds_suffix}.pickle", "rb") as f:
        return pickle.load(f) 

  
def encode_time_value(val, max_val):
    return sin(2* pi * val/max_val), cos(2*pi * val/max_val)

def bulk_encode_time_value(val, max_val):
    x = np.sin(2* np.pi/max_val * val)
    y = np.cos(2*np.pi /max_val * val)
    return np.stack([x,y], axis=1)
        