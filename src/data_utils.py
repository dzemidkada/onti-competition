import pandas as pd
import numpy as np

import os
import yaml
import hashlib

VALIDATION_SOLT = 'GSvsdv[f30-fj[23pmf\'\v,mf]4[0-fj[9q03j[f3vf,;\w,c03jrf23'

def get_client_id_hash(client_id):
    return hashlib.md5(f'{VALIDATION_SOLT}{client_id}'.encode('utf-8')).hexdigest()

def get_hash_bin(hash_hex, bins=10):
    return int(hash_hex, 16) % bins

def client_id_bins(df):
    return pd.DataFrame({
        'client_id': df.client_id.unique(),
        'bin': list(map(get_hash_bin, map(get_client_id_hash, df.client_id.unique())))
    })

def adjust_path(path):
    return os.path.abspath(os.path.join(__file__ , '../..', path))

class DataSource:
    def __init__(self, cfg=None):
        assert cfg is not None, 'Config is empty'
        self._cfg = cfg
        self._data = None
    
    def read_data(self):
        if self._data:
            return
        self._data = dict()
        for key, path in self._cfg['data_source'].items():
            print(f'Reading {key}...')
            self._data[key] = pd.read_csv(adjust_path(path))
    
    def validation_split(self):
        print('Validation split: by clientID')
        train_x = self.get_data('train_x')
        train_target = self.get_data('train_target')
        
        bins_df = client_id_bins(train_target)
        
        # Transactions split
        train_x_bin = train_x.merge(bins_df, how='left', on='client_id')
        train_mask = train_x_bin.bin < 8
        self.set_data('train_x', train_x_bin[train_mask].drop('bin', axis=1).reset_index(drop=True))
        self.set_data('valid_x', train_x_bin[~train_mask].drop('bin', axis=1).reset_index(drop=True))
        
        # Targets split
        train_target_bin = train_target.merge(bins_df, how='left', on='client_id')
        train_mask = train_target_bin.bin < 8
        self.set_data('train_target', train_target_bin[train_mask].drop('bin', axis=1).reset_index(drop=True))
        self.set_data('valid_target', train_target_bin[~train_mask].drop('bin', axis=1).reset_index(drop=True))
    
    def get_data(self, data):
        if self._data is None:
            return None
        return self._data.get(data, None)

    def set_data(self, data, df):
        if df is None:
            return
        self._data[data] = df
    
    def add_features(self, name, path):
        with open(path, 'rb') as f:
            features = np.load(f, allow_pickle = True)
        self.set_data(name, features)
    
    def __str__(self):
        return '\n'.join(
            f'Dataset: {k}, shape: {v.shape}'
            for k, v in self._data.items()
        )
            
            