import os
import yaml

def get_default_config_path():
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), 'cfg.yaml')

class Config:
    def __init__(self, cfg_path=None):
        cfg_path = cfg_path or get_default_config_path()
        self._cfg = yaml.load(open(cfg_path, 'r'), Loader=yaml.FullLoader)
    
    def __getitem__(self, x):
        return self._cfg.get(x, None)