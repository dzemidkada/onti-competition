from functools import partial
import pandas as pd
import numpy as np
import tqdm

PERCENTILES = (10, 33, 66, 90, 95)
HIGH_LEVEL_STATS = (np.min, np.max, np.std, np.mean, np.median, *[partial(np.percentile, q=p) for p in PERCENTILES])

def convert_dates(dates, output=None):
    if output == 'day_of_week':
        return dates % 7
    if output == 'month':
        return (dates // 30) % 12
    if output == 'quarter':
        return (dates // 90) % 4
    return dates

class GlobalClientFeaturesExtractor:
    AGG_LEVELS = (('month', 30), ('quarter', 90), ('year', ))
    def __init__(self, cfg=None):
        self._cfg = cfg
    
    def extract(self, df, client_id):
        transactions = df.query(f'client_id')
        n = transactions.shape[0]
        features = []
        
        ############################################
        # Dates ####################################
        ############################################
        print('Dates features extraction...')
        t_dates = transactions.trans_date.values
        
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
        print('Money features extraction...')
        money = transactions.amount_rur.values
        
        week_day_money_stats = np.array([list(money[dow_dates == day_of_week].describe()) for day_of_week in range(7)]).ravel()
        month_money_stats = np.array([list(money[m_dates == month].describe()) for month in range(12)]).ravel()
        quarter_money_stats = np.array([list(money[q_dates == quarter].describe()) for quarter in range(4)]).ravel()
        
        agg_features = []
        for name, duration in tqdm.tqdm_notebook(AGG_LEVELS):
            print(f'Money features, agg level: {name}')
            buckets = transactions.assign(bucket_id = transactions.trans_date // duration)
            for bucker_id, bucket_df in buckets.groupby(bucket_id):
                bucket_money = bucket_df.amount_rur.values
                for _ in tqdm.tqdm_notebook(HIGH_LEVEL_STATS):
                    agg_features.append(_(bucket_money))
        
        features.extend([
            *list(money.describe()), *list(week_day_money_stats), *list(month_money_stats), *list(quarter_money_stats),
            *agg_features,
            # TODO(dzmr): Add more aggregations
        ])    
        
        # TODO(dzmr): Outliers
        # TODO(dzmr): Groups
        # TODO(dzmr): Clients clustering
        
        return features

        
        