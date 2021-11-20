# this module provide the basic functionality to
# (1) construct stock net according to the features of each stock by calculating the similarities
# (2) provide a back test function
#   (给定一个频率, 按照这个频率按照某个因子进行选股换仓, 其中, 这个因子根据股票的关联网络计算得出;
#   并且可以设定每只股票的持有量, 例如是等手数持有还是等金额持有)

from itertools import combinations
from typing import List

from prepare_data import *
from scipy.spatial.distance import cosine


def get_stock_feature(stock_id: str or List[str],
                      timestamp,
                      feature_list=None
                      ) -> pd.DataFrame:
    """
    在给定的时间点, 获取给定股票的特征, 更聚合的特征(可以反映时间序列的相似度)是: 股票前一个月的交易数据: 具体包括: [股票收盘价的累计涨跌幅,
    平均交易量, ] 当然也不一定要这么麻烦, 可以把股票一个月内的高开低收成交量 5 * 22 个数据按照时间先后排成一个向量, 并且分别除以股票自己这
    个月的第一天的数据,进行归一化.
    :param feature_list:
    :param stock_id:
    :param timestamp:
    :return: 特征数据: 股票的id作为index, 每一行都是股票的特征
    """
    if feature_list is not None:
        raise NotImplementedError
    else:
        pass
    if isinstance(stock_id, str):
        stock_id = [stock_id]
    assert isinstance(stock_id, list) or isinstance(stock_id, np.ndarray)
    date_range = pd.date_range(end=timestamp, freq='1D', periods=30)[:-1]
    _data = stock_data.loc[(stock_data['stock_id'].isin(stock_id)) & (stock_data['timestamp'].isin(date_range)), :]
    _result = []
    for stock in stock_id:
        _d = _data.loc[_data['stock_id'] == stock, ['open', 'high', 'low', 'close', 'volume']].values
        if _d.shape[0] == 0:
            _d = np.ones(shape=(1, 5))
        _scale = _d[0, :]
        _d = _d / _scale
        _d = _d.reshape(-1, 1)
        _result.append(_d)
    lengths = [_.shape[0] for _ in _result]
    max_length = max(lengths)
    for idx, _ in enumerate(_result):
        _result[idx] = pad_to_max_len(_, max_length=max_length, unit=5)

    _result = pd.DataFrame(index=stock_id, data=np.array(_result).squeeze())

    return _result


def pad_to_max_len(array: np.ndarray,
                   max_length: int,
                   unit=5) -> np.ndarray:
    while len(array) < max_length:
        array = np.append(array, array[-unit:])
    return array.reshape(-1, 1)


def get_similarity(features: pd.DataFrame) -> pd.DataFrame:
    """
    根据给定的特征计算给定股票列表两两的相关性
    :param features: 股票的id作为index, 每一行都是股票的特征
    :return: 相关性矩阵: index: stock_list, columns: stock_list
    """
    stocks = features.index
    stocks_combination = combinations(stocks, 2)
    _result = pd.DataFrame(index=stocks, columns=stocks, data=0)
    for i, j in stocks_combination:
        _result.loc[i, j] = cosine(features.loc[i, :], features.loc[j, :])
        _result.loc[j, i] = _result.loc[i, j]
    return _result


def get_stocks_price_change(stock_list, timestamp) -> pd.Series:
    """
    index: stock_list, data: price change percent
    :param stock_list:
    :param timestamp:
    :return:
    """
    _result = get_stock_price(stock_list=stock_list, timestamp=timestamp)
    _result = _result.apply(lambda x: x.iloc[-1] / x.iloc[0] - 1, axis=0)
    return _result


def get_traction_factor(similarity_matrix: pd.DataFrame,
                        price_percent_change: pd.Series,
                        n: int = 10) -> pd.Series:
    """
    直觉上的思路是: 如果一只股票涨了, 那么在这个关联度网络中和它关联度高的股票也会被拉涨, 这个月不涨, 下个月也可能涨, 所以这个因子有点牵引度的意思
    具体的计算思路是: 对于某只股票, 选取与其关联度最高的前 10只股票, 并且乘以 这10只股票过去一个月的累计涨跌幅, 求和
    :param price_percent_change:
    :param similarity_matrix:
    :param n:
    :return:
    """
    # todo: need further inspection
    stocks = similarity_matrix.index
    _weight_list = []
    for i in stocks:
        _weights = similarity_matrix.loc[:, i].nlargest(n=n)
        _weight_list.append(_weights)

    _weights = pd.concat(_weight_list, axis=1)

    _result = _weights * price_percent_change
    _result.dropna(how='all', inplace=True)
    return _result.sum(axis=0)


def pick_stocks(factor_series) -> np.ndarray:
    """
    选取factor_series中因子值最小的10只股票作为持仓
    :param factor_series:
    :return:
    """
    return factor_series.nsmallest(10).index.values


def get_asset_series(stock_list,
                     principal,
                     opening_rule: str,
                     datetime_span: List) -> pd.Series:
    """
    根据持仓手数, 返回一个series: index: 从 datetime_span[0] 到 datetime_span[1], data: 每天的资金量
    :param datetime_span: [start_date, end_date], e.g. ['2016-01-01', '2016-01-31'] 一个月的时间
    :param stock_list: 选择持仓的股票
    :param principal: 本金
    :param opening_rule: '等金额' or '等数量'
    :return:
    """
    # 获得持仓数量
    _opening_price = get_stock_price(stock_list, datetime_span[0])
    _start = pd.to_datetime(datetime_span[0])
    while _opening_price is None:
        _start = _start + pd.to_timedelta('1D')
        _opening_price = get_stock_price(stock_list, _start)

    _quantity_per_stock = get_trading_position(_opening_price, opening_rule, principal, stock_list)

    # 根据持仓数量, 计算每个时刻的资产总量
    _prices = get_stock_price(stock_list, timestamp=datetime_span)
    _result = _prices.mul(_quantity_per_stock, axis=1)
    _result = _result.sum(axis=1)
    # _result.fillna(method='ffill', inplace=True)
    return _result


def get_trading_position(_opening_price: pd.Series or pd.DataFrame,
                         opening_rule: str,
                         principal: float or int,
                         stock_list: List or np.ndarray) -> pd.Series:
    """
    获得每只股票的持仓数量
    :param _opening_price:
    :param opening_rule:
    :param principal:
    :param stock_list: 股票列表
    :return:
    """
    if opening_rule == '等金额':
        _asset_per_stock = principal / len(stock_list)
        _quantity_per_stock = _asset_per_stock / _opening_price
        _quantity_per_stock = _quantity_per_stock.iloc[0, :]
    elif opening_rule == '等数量':
        _quantity_per_stock = principal / _opening_price.sum(axis=1)
        _quantity_per_stock = _quantity_per_stock.values[0]
        _quantity_per_stock = pd.Series(index=stock_list, data=_quantity_per_stock)
    else:
        _quantity_per_stock = pd.Series(index=stock_list, data=0)
    return _quantity_per_stock


def get_stock_price(stock_list,
                    timestamp) -> pd.DataFrame or None:
    """
    返回stock_list中每只股票的在timestamp的收盘价
    :param stock_list:
    :param timestamp:
    :return:
    """
    if isinstance(timestamp, list) or isinstance(timestamp, np.ndarray):
        timestamp = pd.date_range(start=timestamp[0], end=timestamp[1], freq='D')
        _result = stock_data.loc[(stock_data['stock_id'].isin(stock_list)) & (stock_data['timestamp'].isin(timestamp)),
                                 ['close', 'stock_id', 'timestamp']]
    else:
        _result = stock_data.loc[(stock_data['stock_id'].isin(stock_list)) & (stock_data['timestamp'] == timestamp),
                                 ['close', 'stock_id', 'timestamp']]

    # _result.index = _result['timestamp']
    if _result.shape[0] == 0:
        return

    _result = _result.pivot(index='timestamp', columns='stock_id', values='close')

    return _result


def split_date_time(start, end) -> np.ndarray:
    """
    按照月份划分时间, 返回一个 2-dim array, [[month_start, month_end]...]
    :param start:
    :param end:
    :return:
    """
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    a = pd.DataFrame(index=pd.date_range(start, end, freq='1D'), dtype=int)
    a['date'] = a.index
    return a.resample('M')['date'].agg(['first', 'last']).values


def back_test(start,
              end,
              principal: int or float,
              opening_rule: str,
              stock_pool: List) -> pd.Series:
    """

    :param stock_pool:
    :param opening_rule:
    :param principal:
    :param start: 回测开始的时间
    :param end: 回测结束的时间
    :return:
    """
    date_span_list = split_date_time(start, end)
    asset_series_list = []
    _principal = principal
    last_i, last_j = 0, 0
    for idx, (i, j) in enumerate(date_span_list):
        if idx >= 2:
            features = get_stock_feature(stock_id=stock_pool, timestamp=i, feature_list=None)
            similarity_matrix = get_similarity(features=features)
            price_pct_change = get_stocks_price_change(stock_list=stock_pool, timestamp=[last_i, last_j])
            factor_series = get_traction_factor(similarity_matrix=similarity_matrix,
                                                price_percent_change=price_pct_change)
            stock_list = pick_stocks(factor_series=factor_series)
            asset_series = get_asset_series(stock_list=stock_list,
                                            principal=_principal,
                                            opening_rule=opening_rule,
                                            datetime_span=[i, j])
            asset_series_list.append(asset_series)
            _principal = asset_series.iloc[-1]
        last_i, last_j = i, j

    return pd.concat(asset_series_list, axis=0)


def get_return_ratio(asset_series: pd.Series) -> float:
    asset_series.sort_index(inplace=True)
    return asset_series.iloc[-1] / asset_series.iloc[0] - 1


def get_annual_return(asset_series: pd.Series) -> float:
    return get_return_ratio(asset_series) / get_date_span(asset_series) * 365


def get_date_span(asset_series: pd.Series) -> float or int:
    _dates = asset_series.index
    return (_dates.max() - _dates.min()).days


def get_sharpe_ratio(asset_series: pd.Series,
                     freq='daily') -> float:
    tmp = asset_series.shift(-1) / asset_series - 1
    tmp = tmp.dropna()
    if freq == 'daily':
        return tmp.mean() / tmp.std() * np.sqrt(252)


def get_volatility(asset_series: pd.Series, freq='daily'):
    tmp = asset_series.shift(-1) / asset_series - 1
    tmp = tmp.dropna()
    if freq == 'daily':
        return tmp.std() * np.sqrt(252)


def get_max_draw_down(asset_series):
    j = np.argmax((np.maximum.accumulate(asset_series) - asset_series) / asset_series)
    if j == 0:
        return 0
    i = np.argmax(asset_series[:j])
    d = (asset_series[i] - asset_series[j]) / asset_series[i]
    return d




