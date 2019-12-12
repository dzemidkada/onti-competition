from functools import partial
from collections import defaultdict
import pandas as pd
import numpy as np
import tqdm

PERCENTILES = (10, 33, 66, 90, 95)
HIGH_LEVEL_STATS = (np.min, np.max, np.sum, np.std, np.mean, np.median)
OUTLIERS_THRESHOLDS = (500, 5000, 10000, 30000, 50000, 100000, 1000000)

IMPORTANT_GROUPS = (0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18, 147, 20, 21, 23, 24, 25, 26, 151, 28, 29,
                    31, 32, 34, 35, 36, 37, 39, 43, 44, 46, 50, 55, 57, 59, 61, 64, 69, 80, 81, 82, 105, 109, 110, 112, 123)

def convert_dates(dates, output=None):
    if output == 'day_of_week':
        return dates % 7
    if output == 'month':
        return (dates // 30) % 12
    if output == 'quarter':
        return (dates // 90) % 4
    return dates

def get_important_groups(train_data, top_k=30):
    important_groups = set()
    for fa in (np.size, np.max, np.sum):
        group_index = (
            train_x
            .groupby('small_group').amount_rur.agg(fa)
            .sort_values(ascending=False)
            .iloc[:top_k].index.tolist()
        )
        important_groups.update(group_index)
    return list(important_groups)

class GlobalClientFeaturesExtractor:
    
    AGG_LEVELS = (('month', 30), ('quarter', 90))
    
    def __init__(self, cfg=None):
        self._cfg = cfg
    
    def extract(self, transactions, client_id):
        n = transactions.shape[0]
        features = []
        
        ############################################
        # Dates ####################################
        ############################################
        #print('Dates features extraction...')
        t_dates = transactions.trans_date
        
        unique_dates = t_dates.nunique()
        period_length = t_dates.max() - t_dates.min() + 1
        
        # fraction of transactions for a week day
        dow_dates = convert_dates(t_dates, 'day_of_week')
        week_day_fracs = np.array([(dow_dates == day_of_week).sum() / n for day_of_week in range(7)])
        # fraction of transactions for a month
        m_dates = convert_dates(t_dates, 'month')
        month_fracs = np.array([(m_dates == month).sum() / n for month in range(12)])
        # fraction of transactions for a quarter
        q_dates = convert_dates(t_dates, 'quarter')
        quarter_fracs = np.array([(q_dates == quarter).sum() / n for quarter in range(4)])
        transactions_per_day_info = transactions.groupby('trans_date').trans_date.count().describe()
        
        features.extend([
            period_length,
            unique_dates / period_length, # prob of [>=1] transactions for a given day
            *list(week_day_fracs), week_day_fracs.mean(), week_day_fracs.std(), week_day_fracs.min(), week_day_fracs.max(),
            *list(month_fracs), month_fracs.mean(), month_fracs.std(), month_fracs.min(), month_fracs.max(),
            *list(quarter_fracs), quarter_fracs.mean(), quarter_fracs.std(), quarter_fracs.min(), quarter_fracs.max(),
            *list(transactions_per_day_info)
        ])
        
        ############################################
        # Money ####################################
        ############################################
        #print('Money features extraction...')
        money = transactions.amount_rur
        
        week_day_money_stats = np.array([list(money[dow_dates == day_of_week].describe()) for day_of_week in range(7)]).ravel()
        month_money_stats = np.array([list(money[m_dates == month].describe()) for month in range(12)]).ravel()
        quarter_money_stats = np.array([list(money[q_dates == quarter].describe()) for quarter in range(4)]).ravel()
        
        agg_features = []
        for name, duration in self.AGG_LEVELS:
            #print(f'Money features, agg level: {name}')
            buckets = transactions.assign(bucket_id = transactions.trans_date // duration)
            buckets_values = defaultdict(list)
            for bucker_id, bucket_df in buckets.groupby('bucket_id'):
                bucket_money = bucket_df.amount_rur.values
                for fa in HIGH_LEVEL_STATS:
                    #print(fa.__name__)
                    value = fa(bucket_money)
                    agg_features.append(value)
                    buckets_values[fa.__name__].append(value)
            for fa in HIGH_LEVEL_STATS:
                values = np.array(buckets_values[fa.__name__])
                for fb in HIGH_LEVEL_STATS:
                    agg_features.append(fb(value))
                
        # Outliers
        outliers_features = []
        for threshold in OUTLIERS_THRESHOLDS:
            #print(f'Outliers threshold: {threshold}')
            outliers = transactions.assign(is_outlier = transactions.amount_rur > threshold)
            for fa in (np.min, np.max, np.sum):
                outliers_features.append(fa(outliers.is_outlier))
                outliers_features.append(fa(outliers.query('is_outlier == True').amount_rur))
                
        features.extend([
            *list(money.describe()), *list(week_day_money_stats), *list(month_money_stats), *list(quarter_money_stats),
            *agg_features,
            *outliers_features,
        ])
        
        ############################################
        # Groups ###################################
        ############################################
        
        # Just one-hot count, amount sum, amount max
        groups_skeleton = pd.DataFrame({'small_group': IMPORTANT_GROUPS})
        
        group_features = []
        for fa in (np.size, np.sum, np.max):
            g_features = (
                pd.merge(groups_skeleton,
                         transactions.groupby('small_group').amount_rur.agg(fa),
                         on='small_group', how='left')
                .amount_rur.tolist()
            )
            group_features.extend(g_features)
            
        features.extend([
            *group_features,
        ])
        
        return np.nan_to_num(features)