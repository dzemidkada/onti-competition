import pandas as pd
import numpy as np

class GlobalClientFeaturesExtractor:
    def __init__(self, cfg):
        self._cfg = cfg
    
    def extract(self, df):
        
        