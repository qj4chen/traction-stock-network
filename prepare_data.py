import os
import time

import numpy as np
import pandas as pd
import tushare as ts

pro = ts.pro_api(token='6a721053ea3e70bb52605d6c0972caeda9ff080d3671f69bd8b6b434')

if not os.path.exists('./data/stock_list.csv'):
    pro.stock_basic(list_status='L',
                    fields='ts_code, symbol, name, area, industry, fullname, market, list_date, delist_date').to_csv(
        './data/stock_list.csv')

STOCK_LIST = pd.read_csv('./data/stock_list.csv', index_col=0)

# 调取过去 10年的股票行情数据
start_date = '2011-11-01'
end_date = '2021-11-01'
stock_data_list = []
STOCK_LIST = STOCK_LIST.loc[(STOCK_LIST['list_date'] > 20000101) & (STOCK_LIST['list_date'] < 20111101) & (
            STOCK_LIST['market'] == '主板'), 'ts_code'].values
STOCK_LIST = [s for s in STOCK_LIST if s.endswith('SH')]
for s in STOCK_LIST:
    filename = f'./data/daily_k_lines_{s.replace(".", "_")}_{start_date}_{end_date}.csv'
    if not os.path.exists(filename):
        pro.daily(ts_code=s, start_date=start_date, end_date=end_date).to_csv(filename)
        time.sleep(0.05)

    filename = f'./data/daily_k_lines_{s.replace(".", "_")}_{start_date}_{end_date}.csv'
    stock_data_list.append(pd.read_csv(filename, index_col=0))

stock_data = pd.concat(stock_data_list, axis=0)
stock_data.reset_index(inplace=True)
stock_data = stock_data[['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol']]
stock_data.rename({'ts_code': 'stock_id', 'vol': 'volume', 'trade_date': 'timestamp'}, inplace=True, axis=1)
stock_data['timestamp'] = stock_data['timestamp'].astype(str)
stock_data['timestamp'] = pd.to_datetime(stock_data['timestamp'])
