import argparse
import copy
import pickle

import numpy as np
import os
import random

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from time import time

from matplotlib import pyplot as plt

from demo.data_util import FACTORDATA
from demo.index_member import get_index_member
from load_data import load_EOD_data, load_relation_data, load_factors
from demo.config import Config

seed = 123456789
np.random.seed(seed)
torch.manual_seed(seed)


class LeakyReLU(nn.Module):
    def __init__(self, alpha: float = .2):
        super(LeakyReLU, self).__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(self.alpha * x, x)


class ReRaLSTM(nn.Module):

    def __init__(
            self,
            parameters,
            tickers: list[str],
            factor: list[str],
            dates: list[str],
            relation_path: str,
            epochs: int = 50,
            steps: int = 1,
            batch_size=None,
            gpu: bool = False,
            flat: bool = False,
            in_pro: bool = False,
            early_stop: bool = True,
            y_label: str = 'price',  # 'price' or 'rtn'

    ):
        super(ReRaLSTM, self).__init__()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
        self.tickers = tickers
        self.steps = steps
        self.epochs = epochs
        self.params = copy.deepcopy(parameters)

        print('# tickers selected:', len(self.tickers))
        self.eod_data, self.mask_data, self.gt_data, self.price_data = load_EOD_data(
            self.tickers,
            dates,
            standard=True,
            steps=self.steps
        )
        if factor is not None:
            self.factor_data, self.factor_mask = load_factors(factors=factor,
                                                              tickers=self.tickers, trading_days=dates, standard=True)
            self.eod_data = np.concatenate((self.eod_data, self.factor_data), axis=2)
            self.mask_data = self.mask_data * self.factor_mask
        self.rel_encoding, self.rel_mask = load_relation_data(os.path.join(Config.PATH, relation_path), self.tickers)
        print('relation mask shape:', self.rel_mask.shape)
        # self.embedding = load_hs_embedding(os.path.join(Config.PATH,embedding_path),tickers=self.tickers,trading_days=dates)
        # print('embedding shape:', self.embedding.shape)
        self.trade_dates = self.mask_data.shape[1]  # int
        self.input_dim = self.eod_data.shape[2]  # 因子个数
        self.hidden_dim = self.params['unit']
        self.flat = flat
        self.inner_prod = in_pro
        self.early_stop = early_stop
        self.y_label = y_label
        self.batch_size = len(self.tickers) if batch_size is None else batch_size
        self.valid_index = int(self.eod_data.shape[1] * 0.7)
        self.test_index = int(self.eod_data.shape[1] * 0.85)
        self.criterion = nn.MSELoss()
        self.leaky_relu = LeakyReLU(0.2).to(self.device)
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True).to(self.device)
        # self.bn1 = nn.BatchNorm1d(self.hidden_dim).to(self.device)  # 添加BatchNorm
        self.dense1 = nn.Linear(self.rel_encoding.shape[-1], 1).to(self.device)  # 更新的是dense的权重
        self.dense2 = nn.Linear(self.params['unit'], 1).to(self.device)
        self.dense3 = nn.Linear(self.params['unit'], 1).to(self.device)
        self.dense4 = nn.Linear(self.params['unit'] * 2, 1).to(self.device)
        self.dense5 = nn.Linear(self.params['unit'] * 2, self.params['unit']*2).to(self.device) if flat else None

    def get_batch(self, offset=None):
        if offset is None:
            offset = random.randrange(0, self.valid_index)
            print(offset)

        seq_len = self.params['seq']
        mask_batch = self.mask_data[:, offset: offset + seq_len + self.steps - 1]
        mask_batch = np.min(mask_batch, axis=1)
        return (
            self.eod_data[:, offset:offset + seq_len, :],
            np.expand_dims(mask_batch, axis=1),
            np.expand_dims(self.price_data[:, offset + seq_len - 1], axis=1),
            np.expand_dims(self.gt_data[:, offset + seq_len + self.steps - 1], axis=1),  # step日后的收益
        )

    def forward(self, x, relation, rel_mask):
        lstm_out, _ = self.lstm(x)
        seq_emb = lstm_out[:, -1, :]
        # seq_emb = self.bn1(seq_emb)  # normalized_sequence embedding
        rel_weight = self.leaky_relu(self.dense1(relation))  # N*N*10（十个类别）加权为N*N*1
        if self.inner_prod:
            inner_weight = torch.matmul(seq_emb, seq_emb.transpose(1, 0))
            weight = inner_weight * rel_weight[:, :, -1]
        else:
            head_weight = self.leaky_relu(self.dense2(seq_emb))
            tail_weight = self.leaky_relu(self.dense3(seq_emb))
            weight = head_weight + tail_weight + rel_weight[:, :, -1]

        weight_masked = torch.softmax(rel_mask + weight, dim=0)
        outputs_proped = torch.matmul(weight_masked, seq_emb)

        if self.flat:
            # print('one more hidden layer')
            outputs_concated = self.leaky_relu(self.dense5(torch.cat([seq_emb, outputs_proped], dim=1)))
        else:
            outputs_concated = torch.cat([seq_emb, outputs_proped], dim=1)

        prediction = self.leaky_relu(self.dense4(outputs_concated))
        return prediction

    @staticmethod
    def compute_rank_loss(return_ratio, ground_truth, mask):
        pre_pw_dif = return_ratio - return_ratio.T
        gt_pw_dif = ground_truth - ground_truth.T
        mask_pw = mask * mask.T
        rank_loss = torch.mean(torch.relu(pre_pw_dif * gt_pw_dif * mask_pw))
        return rank_loss

    def compute_loss(self, prediction: torch.tensor, base_price: torch.tensor, mask: torch.tensor,
                     ground_truth: torch.tensor):
        if self.y_label == 'price':
            gt_prediction = (prediction - base_price) / base_price
        elif self.y_label == 'rtn':
            gt_prediction = prediction
        else:
            raise ValueError('y_label must be "price" or "rtn"')
        reg_loss = self.criterion(gt_prediction * mask, ground_truth * mask)  # 收益率回归的loss
        rank_loss = self.compute_rank_loss(gt_prediction, ground_truth, mask)  # 根据损失函数调整mask_pw
        loss = reg_loss + self.params['alpha'] * rank_loss  # 后一部分是排名的损失
        return loss, reg_loss, rank_loss

    def train_model(self, early_stop_epoch: int = 10):

        best_valid_perf = {
            'mse': np.inf,
            'mrrt': .0,
            'btl': .0,
        }
        best_model_list = None
        best_valid_mask = None
        best_epoch = 0
        best_test_perf = {
            'mse': np.inf,
            'mrrt': .0,
            'btl': .0,
        }
        relation = torch.tensor(self.rel_encoding, dtype=torch.float32).to(self.device)
        rel_mask = torch.tensor(self.rel_mask, dtype=torch.float32).to(self.device)

        optimizer = optim.Adam(self.parameters(), lr=self.params['lr'])
        best_valid_pred = np.zeros([len(self.tickers), self.test_index - self.valid_index], dtype=float)
        best_valid_gt = np.zeros([len(self.tickers), self.test_index - self.valid_index], dtype=float)
        best_test_pred = np.zeros([
            len(self.tickers),
            self.trade_dates - self.params['seq'] - self.test_index - self.steps + 1,
        ], dtype=float)
        best_test_gt = np.zeros([
            len(self.tickers),
            self.trade_dates - self.params['seq'] - self.test_index - self.steps + 1,
        ], dtype=float)
        best_test_mask = np.zeros([
            len(self.tickers),
            self.trade_dates - self.params['seq'] - self.test_index - self.steps + 1,
        ], dtype=float)
        best_valid_loss = np.inf
        batch_offsets = np.arange(start=0, stop=self.valid_index, dtype=int)

        for epoch in range(self.epochs):
            t1 = time()
            np.random.shuffle(batch_offsets)
            tra_loss = .0
            tra_reg_loss = .0
            tra_rank_loss = .0

            for j in range(self.valid_index - self.params['seq'] - self.steps + 1):
                eod_batch, mask_batch, price_batch, gt_batch = self.get_batch(batch_offsets[j])
                eod = torch.tensor(eod_batch, dtype=torch.float32).to(self.device)
                mask = torch.tensor(mask_batch, dtype=torch.float32).to(self.device)
                ground_truth = torch.tensor(gt_batch, dtype=torch.float32).to(self.device)
                base_price = torch.tensor(price_batch, dtype=torch.float32).to(self.device)

                prediction = self.forward(eod, relation, rel_mask)  # 这里嵌入relation，feature为e，

                loss, reg_loss, rank_loss = self.compute_loss(prediction, base_price, mask, ground_truth)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tra_loss += loss.item()
                tra_reg_loss += reg_loss.item()
                tra_rank_loss += rank_loss.item()

            print(
                'Train Loss:',
                tra_loss / (self.valid_index - self.params['seq'] - self.steps + 1),
                tra_reg_loss / (self.valid_index - self.params['seq'] - self.steps + 1),
                tra_rank_loss / (self.valid_index - self.params['seq'] - self.steps + 1),
            )

            val_loss = .0
            val_reg_loss = .0
            val_rank_loss = .0
            cur_valid_pred = np.zeros([len(self.tickers), self.test_index - self.valid_index], dtype=float)
            cur_valid_gt = np.zeros([len(self.tickers), self.test_index - self.valid_index], dtype=float)
            cur_valid_mask = np.zeros([len(self.tickers), self.test_index - self.valid_index], dtype=float)
            for cur_offset in range(
                    self.valid_index - self.params['seq'] - self.steps + 1,
                    self.test_index - self.params['seq'] - self.steps + 1,
            ):
                eod_batch, mask_batch, price_batch, gt_batch = self.get_batch(cur_offset)
                eod = torch.tensor(eod_batch, dtype=torch.float32).to(self.device)
                mask = torch.tensor(mask_batch, dtype=torch.float32).to(self.device)
                ground_truth = torch.tensor(gt_batch, dtype=torch.float32).to(self.device)
                base_price = torch.tensor(price_batch, dtype=torch.float32).to(self.device)

                prediction = self.forward(eod, relation, rel_mask)  # 这里嵌入relation，feature为e，
                loss, reg_loss, rank_loss = self.compute_loss(prediction, base_price, mask, ground_truth)

                val_loss += loss.item()
                val_reg_loss += reg_loss.item()
                val_rank_loss += rank_loss.item()
                cur_valid_pred[:, cur_offset - (self.valid_index - self.params[
                    'seq'] - self.steps + 1)] = prediction.cpu().detach().numpy()[:, 0]
                cur_valid_gt[:, cur_offset - (self.valid_index - self.params[
                    'seq'] - self.steps + 1)] = ground_truth.cpu().detach().numpy()[:, 0]
                cur_valid_mask[:,
                cur_offset - (self.valid_index - self.params['seq'] - self.steps + 1)] = mask.cpu().detach().numpy()[:,
                                                                                         0]

            print(
                'Valid Loss:',
                val_loss / (self.test_index - self.valid_index),
                val_reg_loss / (self.test_index - self.valid_index),
                val_rank_loss / (self.test_index - self.valid_index),
            )
            # cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt, cur_valid_mask)
            # print('\t Valid performance:', cur_valid_perf)

            test_loss = .0
            test_reg_loss = .0
            test_rank_loss = .0
            cur_test_pred = np.zeros([len(self.tickers), self.trade_dates - self.test_index], dtype=float)
            cur_test_gt = np.zeros([len(self.tickers), self.trade_dates - self.test_index], dtype=float)
            cur_test_mask = np.zeros([len(self.tickers), self.trade_dates - self.test_index], dtype=float)
            for cur_offset in range(
                    self.test_index - self.params['seq'] - self.steps + 1,
                    self.trade_dates - self.params['seq'] - self.steps + 1,
            ):
                eod_batch, mask_batch, price_batch, gt_batch = self.get_batch(cur_offset)
                eod = torch.tensor(eod_batch, dtype=torch.float32).to(self.device)
                mask = torch.tensor(mask_batch, dtype=torch.float32).to(self.device)
                ground_truth = torch.tensor(gt_batch, dtype=torch.float32).to(self.device)
                base_price = torch.tensor(price_batch, dtype=torch.float32).to(self.device)
                prediction = self.forward(eod, relation, rel_mask)  # 这里嵌入relation，feature为e，
                loss, reg_loss, rank_loss = self.compute_loss(prediction, base_price, mask, ground_truth)

                test_loss += loss.item()
                test_reg_loss += reg_loss.item()
                test_rank_loss += rank_loss.item()
                cur_test_pred[:, cur_offset - (
                            self.test_index - self.params['seq'] - self.steps + 1)] = prediction.cpu().detach().numpy()[
                                                                                      :, 0]
                cur_test_gt[:, cur_offset - (self.test_index - self.params[
                    'seq'] - self.steps + 1)] = ground_truth.cpu().detach().numpy()[:, 0]
                cur_test_mask[:,
                cur_offset - (self.test_index - self.params['seq'] - self.steps + 1)] = mask.cpu().detach().numpy()[:,
                                                                                        0]

            print(
                'Test Loss:',
                test_loss / (self.trade_dates - self.test_index),
                test_reg_loss / (self.trade_dates - self.test_index),
                test_rank_loss / (self.trade_dates - self.test_index),
            )
            # cur_test_perf = evaluate(cur_test_pred, cur_test_gt, cur_test_mask)
            # print('\t Test performance:', cur_test_perf)

            if val_loss / (self.test_index - self.valid_index) < best_valid_loss:
                best_valid_loss = val_loss / (self.test_index - self.valid_index)
                # best_valid_perf = copy.deepcopy(cur_valid_perf)
                best_valid_gt = copy.deepcopy(cur_valid_gt)
                best_valid_pred = copy.deepcopy(cur_valid_pred)
                best_valid_mask = copy.deepcopy(cur_valid_mask)
                # best_test_perf = copy.deepcopy(cur_test_perf)
                best_test_gt = copy.deepcopy(cur_test_gt)
                best_test_pred = copy.deepcopy(cur_test_pred)
                best_test_mask = copy.deepcopy(cur_test_mask)
                best_model_list = [copy.deepcopy(self.lstm), copy.deepcopy(self.dense1), copy.deepcopy(self.dense2),
                                   copy.deepcopy(self.dense3), copy.deepcopy(self.dense4), copy.deepcopy(self.dense5)]
                best_epoch = epoch
            print('Better valid loss:', best_valid_loss)
            t4 = time()
            print('epoch:', epoch, ('time: %.4f ' % (t4 - t1)))
            if self.early_stop:
                if epoch - best_epoch >= early_stop_epoch:
                    print(
                        f'Early stop at epoch:{epoch}, best validation loss:{best_valid_loss} with epoch:{best_epoch}')
                    break
            # print('\nBest Valid performance:', best_valid_perf)
            # print('\tBest Test performance:', best_test_perf)

        return best_model_list

    def predict_all_samples(self, model_set: list[nn.Module]):
        relation = torch.tensor(self.rel_encoding, dtype=torch.float32).to(self.device)
        rel_mask = torch.tensor(self.rel_mask, dtype=torch.float32).to(self.device)
        all_predictions = []
        best_lstm, best_dense1, best_dense2, best_dense3, best_dense4, best_dense5 = tuple(model_set)
        with torch.no_grad():
            for offset in range(self.trade_dates - self.params['seq'] - self.steps + 1):
                eod_batch, mask_batch, price_batch, gt_batch = self.get_batch(offset)
                eod = torch.tensor(eod_batch, dtype=torch.float32).to(self.device)
                lstm_out, _ = best_lstm(eod)
                seq_emb = lstm_out[:, -1, :]
                rel_weight = self.leaky_relu(best_dense1(relation))  # 20*20*10（十个类别）加权为20*20*1
                if self.inner_prod:
                    inner_weight = torch.matmul(seq_emb, seq_emb.transpose(1, 0))
                    weight = inner_weight * rel_weight[:, :, -1]
                else:
                    head_weight = self.leaky_relu(best_dense2(seq_emb))
                    tail_weight = self.leaky_relu(best_dense3(seq_emb))
                    weight = head_weight + tail_weight + rel_weight[:, :, -1]

                weight_masked = torch.softmax(rel_mask + weight, dim=0)
                outputs_proped = torch.matmul(weight_masked, seq_emb)

                if self.flat:
                    # print('one more hidden layer')
                    outputs_concated = self.leaky_relu(best_dense5(torch.cat([seq_emb, outputs_proped], dim=1)))
                else:
                    outputs_concated = torch.cat([seq_emb, outputs_proped], dim=1)

                prediction = self.leaky_relu(best_dense4(outputs_concated))
                all_predictions.append(prediction.cpu().numpy())

        return np.concatenate(all_predictions, axis=1)

    def next_embedding(self, model_set: list[nn.Module]):
        relation = torch.tensor(self.rel_encoding, dtype=torch.float32).to(self.device)
        rel_mask = torch.tensor(self.rel_mask, dtype=torch.float32).to(self.device)
        all_output = []
        best_lstm, best_dense1, best_dense2, best_dense3, best_dense4, best_dense5 = tuple(model_set)
        with torch.no_grad():
            for offset in range(self.trade_dates - self.params['seq'] - self.steps + 1):
                eod_batch, mask_batch, price_batch, gt_batch = self.get_batch(offset)
                eod = torch.tensor(eod_batch, dtype=torch.float32).to(self.device)
                lstm_out, _ = best_lstm(eod)
                seq_emb = lstm_out[:, -1, :]
                rel_weight = self.leaky_relu(best_dense1(relation))  # 20*20*10（十个类别）加权为20*20*1
                if self.inner_prod:
                    inner_weight = torch.matmul(seq_emb, seq_emb.transpose(1, 0))
                    weight = inner_weight * rel_weight[:, :, -1]
                else:
                    head_weight = self.leaky_relu(best_dense2(seq_emb))
                    tail_weight = self.leaky_relu(best_dense3(seq_emb))
                    weight = head_weight + tail_weight + rel_weight[:, :, -1]

                weight_masked = torch.softmax(rel_mask + weight, dim=0)
                outputs_proped = torch.matmul(weight_masked, seq_emb)

                if self.flat:
                    # print('one more hidden layer')
                    outputs_concated = self.leaky_relu(best_dense5(torch.cat([seq_emb, outputs_proped], dim=1)))
                else:
                    outputs_concated = torch.cat([seq_emb, outputs_proped], dim=1)

                all_output.append(outputs_concated.cpu().numpy())

        return np.array(all_output)


if __name__ == '__main__':
    parameters = {
        'seq': int(4),
        'unit': int(64),
        'lr': float(0.001),
        'alpha': float(1),
    }
    FACTORDATA.DEFAULT_DATA_PATH = "../../data/index_member"
    training_date = [str(date) for date in
                     FACTORDATA(data_path="../../data/index_member").tradingday('20200820', '20240820')]
    # use_tickers = get_index_member(Config.BETA_TICKER, training_date[0].replace('-', ''), training_date[-1].replace('-', ''), include_index=False)
    use_tickers = pd.read_csv('../../results/alpha191_001.csv', index_col=0).columns.tolist()
    factor_list = [str(i) if i >= 100 else '0' + str(i) if i >= 10 else '00' + str(i) for i in range(1, 191)]
    for factor in ['030','017','022','028','039','033']:
        if factor in factor_list:
            factor_list.remove(factor)
    RR_LSTM = ReRaLSTM(  # 所以还是因子的问题！
        parameters=parameters,
        steps=1,
        epochs=50,
        batch_size=None,
        gpu=True,
        in_pro=False,
        dates=training_date,
        tickers=use_tickers,
        factor=factor_list,
        relation_path='relation_embedding_dict.pkl',
        early_stop=True,
        y_label='rtn',
        flat=False

    )
    best_model_set = RR_LSTM.train_model()
    predict = RR_LSTM.predict_all_samples(best_model_set)
    real = RR_LSTM.gt_data[:, 4:]

    compare_array = np.vstack((real[4, :], predict[4, :]))
    compare_df = pd.DataFrame(compare_array, index=['real', 'predict'], columns=pd.to_datetime(training_date)[4:]).T
    compare_df.plot()
    plt.show()

    df_real =  pd.DataFrame(real)
    df_predict = pd.DataFrame(predict)
    ic = df_real.corrwith(df_predict, method='spearman')
    ic.mean()
    ic.cumsum().plot()
    plt.show()
    m=pd.DataFrame(index=training_date[4:],columns=use_tickers)

    factor_df = pd.DataFrame(index=training_date[4:],data=predict.T,columns=use_tickers)
    # factor_df.to_csv('../../results/predicted_price.csv',encoding='gbk')



    relation_lstm_embedding_rtn = RR_LSTM.next_embedding(best_model_set)
    with open(os.path.join(Config.PATH, 'relation_lstm_embedding_rtn.pkl'), 'wb') as f:
        pickle.dump(relation_lstm_embedding_rtn, f)

