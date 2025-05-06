import copy
import pickle
from typing import List

import numpy as np
import os
import pandas as pd
from demo.index_member import get_index_member
from demo.factors.kline_adj import FactorFactoryKlineAdj
from demo.data_util import FACTORDATA
from demo.config import Config
from datetime import datetime
from tqdm import tqdm
Config.PATH = '../../data'
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler



def standardize_matrix(M:np.array,method:str='quantile')->np.array:  # quantile or zscore
    N, T, F = M.shape
    standardized_M = np.copy(M)
    scaler = StandardScaler()
    qt = QuantileTransformer(output_distribution='normal', random_state=42)
    for f in range(F):
        for t in range(T):
            factor_data = M[:, t, f]
            valid_data = factor_data[factor_data != 1e-10]

            if len(valid_data) > 0:

                if method == 'quantile':
                    standardized_values = qt.fit_transform(valid_data.reshape(-1, 1)).reshape(-1)
                else:

                    standardized_values = scaler.fit_transform(valid_data.reshape(-1, 1)).reshape(-1)

                standardized_M[:, t, f][factor_data != 1e-10] = standardized_values

    return standardized_M




def load_EOD_data(tickers: List[str], trading_days: List[str], standard: bool = True, steps: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    trading_days = pd.to_datetime(trading_days)
    eod_data = []
    masks = []
    ground_truth = []
    base_price =[]

    for index, ticker in tqdm(enumerate(tickers)):
        ticker_data = FactorFactoryKlineAdj().compute_impl(ticker)
        ticker_data = ticker_data.loc[trading_days[0]:trading_days[-1]]
        single_EOD = pd.DataFrame(index=trading_days, columns=ticker_data.columns)
        single_EOD.loc[ticker_data.index, :] = ticker_data

        if index == 0:
            print('single EOD data shape:', single_EOD.shape)
            eod_data = np.zeros((len(tickers), len(trading_days), single_EOD.shape[1]), dtype=np.float32)
            masks = np.ones((len(tickers), len(trading_days)), dtype=np.float32)
            ground_truth = np.zeros((len(tickers), len(trading_days)), dtype=np.float32)
            base_price = np.zeros((len(tickers), len(trading_days)), dtype=np.float32)

        mask = single_EOD.isna()
        single_EOD.values[mask.values] = 1e-10
        masks[index, :] = ~mask['ClosePx'].values


        valid_rows = ~mask['ClosePx'].values
        ground_truth[index, steps:] = np.where(valid_rows[steps:],
                                               (single_EOD['ClosePx'].values[steps:] - single_EOD['ClosePx'].values[:-steps]) / single_EOD['ClosePx'].values[:-steps],
                                               0)

        eod_data[index, :, :] = single_EOD.values
        base_price[index, :] = single_EOD['ClosePx'].values

    ground_truth[ground_truth > 1] = 0

    if standard:
        std_eod_data = standardize_matrix(eod_data,'zscore')
        std_eod_data[std_eod_data == 1e-10] = 1.1
        std_eod_data[std_eod_data == 0] = 1e-10
        return std_eod_data, masks, ground_truth, base_price
    else:
        return eod_data, masks, ground_truth, base_price





def load_factors(factors: List[str], tickers: List[str], trading_days: List[str], standard: bool = True) -> tuple[
    np.ndarray, np.ndarray]:
    factor_data = []
    masks = []
    trading_days = pd.to_datetime(trading_days)

    for index, factor in tqdm(enumerate(factors)):
        factor_df = pd.read_csv(os.path.join('../../results', 'alpha191_' + factor + '.csv'), index_col=0,
                                parse_dates=True)
        factor_df = factor_df.loc[trading_days[0]:trading_days[-1], factor_df.columns.isin(tickers)]

        single_factor = pd.DataFrame(index=trading_days, columns=tickers)
        single_factor.loc[factor_df.index, :] = factor_df

        if index == 0:
            print('single factor data shape:', single_factor.shape)
            factor_data = np.zeros((len(tickers), len(trading_days), len(factors)), dtype=np.float32)
            masks = np.ones((len(tickers), len(trading_days)), dtype=np.float32)

        mask = single_factor.isna() | single_factor.map(np.isinf)

        single_factor[mask] = 1e-10
        masks[:, :] = masks[:, :] * (~mask.values.T).astype(int)

        factor_data[:, :, index] = single_factor.T.values

    if standard:
        factor_data = standardize_matrix(factor_data,method='zscore')
        factor_data[factor_data == 1e-10] = 1.1

    return factor_data, masks


def load_graph_relation_data(relation_file,tickers:list[str], lap=False)->np.array:
    with open(relation_file, 'rb') as f:
        embedding_dict = pickle.load(f)
    relation_embedding = embedding_dict['relation_embedding']
    all_tickers = embedding_dict['tickers']
    use_ticker_pos = [i for i in range(len(all_tickers)) if all_tickers[i] in tickers]
    relation_embedding = relation_embedding[np.ix_(use_ticker_pos, use_ticker_pos)]
    print('relation embedding shape:', relation_embedding.shape)
    rel_shape = [relation_embedding.shape[0], relation_embedding.shape[1]]
    mask_flags = np.equal(np.zeros(rel_shape, dtype=int),
                          np.sum(relation_embedding, axis=2))
    ajacent = np.where(mask_flags, np.zeros(rel_shape, dtype=float),
                       np.ones(rel_shape, dtype=float))
    degree = np.sum(ajacent, axis=0)
    for i in range(len(degree)):
        degree[i] = 1.0 / degree[i]
    np.sqrt(degree, degree)
    deg_neg_half_power = np.diag(degree)
    if lap:
        return np.identity(ajacent.shape[0], dtype=float) - np.dot(
            np.dot(deg_neg_half_power, ajacent), deg_neg_half_power)
    else:
        return np.dot(np.dot(deg_neg_half_power, ajacent), deg_neg_half_power)


def load_relation_data(relation_file,tickers:list[str])->tuple[np.array,np.array]:
    with open(relation_file, 'rb') as f:
        embedding_dict = pickle.load(f)
    relation_embedding = embedding_dict['relation_embedding']
    all_tickers = embedding_dict['tickers']
    use_ticker_pos = [i for i in range(len(all_tickers)) if all_tickers[i] in tickers]
    relation_embedding = relation_embedding[np.ix_(use_ticker_pos, use_ticker_pos)]
    print('relation embedding shape:', relation_embedding.shape)

    rel_shape = [relation_embedding.shape[0], relation_embedding.shape[1]]
    mask_flags = np.equal(np.zeros(rel_shape, dtype=int),
                          np.sum(relation_embedding, axis=2))
    mask = np.where(mask_flags, np.ones(rel_shape) * -1e9, np.zeros(rel_shape))
    return relation_embedding, mask

def load_hs_embedding(hs_embedding_path:str,tickers:List[str],trading_days:List[str])->np.array:  # 要求dict有三个元素，1是ticker，2是dates,3是hs_embedding（为N*T*U)，
    # N=len(ticker),一一对应，T为len(dates),U为隐藏层神经元数目
    with open(hs_embedding_path, 'rb') as f:
         embedding_dict = pickle.load(f)
    all_tickers = embedding_dict['tickers']
    all_dates = embedding_dict['dates']
    use_ticker_pos = [i for i in range(len(all_tickers)) if all_tickers[i] in tickers]
    use_date_pos = [i for i in range(len(all_dates)) if
                    all_dates[i] in trading_days]
    return embedding_dict['hs_embedding'][np.ix_(use_ticker_pos, use_date_pos)]


#
# def build_SFM_data(data_path, market_name, tickers):
#     eod_data = []
#     for index, ticker in enumerate(tickers):
#         single_EOD = np.genfromtxt(
#             os.path.join(data_path, market_name + '_' + ticker + '_1.csv'),
#             dtype=np.float32, delimiter=',', skip_header=False
#         )
#         if index == 0:
#             print('single EOD data shape:', single_EOD.shape)
#             eod_data = np.zeros([len(tickers), single_EOD.shape[0]],
#                                 dtype=np.float32)
#
#         for row in range(single_EOD.shape[0]):
#             if abs(single_EOD.iloc[row][-1] + 1234) < 1e-8:
#                 # handle missing data
#                 if row < 3:
#                     # eod_data[index, row] = 0.0
#                     for i in range(row + 1, single_EOD.shape[0]):
#                         if abs(single_EOD.iloc[i][-1] + 1234) > 1e-8:
#                             eod_data[index][row] = single_EOD.iloc[i][-1]
#                             # print(index, row, i, eod_data[index][row])
#                             break
#                 else:
#                     eod_data[index][row] = np.sum(
#                         eod_data[index, row - 3:row]) / 3
#                     # print(index, row, eod_data[index][row])
#             else:
#                 eod_data[index][row] = single_EOD.iloc[row][-1]
#         # print('test point')
#     np.save(market_name + '_sfm_data', eod_data)

if __name__ == '__main__':
    tickers = get_index_member('000300.SH', '20200101', '20230101', include_index=True)[:10]
    trading_days =  [datetime.strptime(str(d), '%Y%m%d').strftime('%Y-%m-%d') for d in FACTORDATA().tradingday('20200101', '20230101')]
    # a, b, c, d = load_EOD_data(tickers,trading_days)
    k = load_graph_relation_data(os.path.join(Config.PATH,'relation_embedding_dict.pkl'),tickers)
    # e,f=load_factors(['049','050','051'],tickers,trading_days)
    #
