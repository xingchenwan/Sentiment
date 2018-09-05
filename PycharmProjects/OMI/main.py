import networkx as nx

from source.utils import *
from source.all_classes import *
import matplotlib.pylab as plt
from collections import Counter
import pandas as pd


def fix_fulldata(full_data):
    """
    A temporary patch to resolve the incompatibility issue between Python 2.x and 3.x pickling
    :param full_data:
    :return:
    """

    def fix_object(obj):
        """
        Decode the byte data type back to strings due to Python 3.x un-pickling
        :param obj: all_data object
        :return: None
        """
        obj.__dict__ = dict((k.decode("ascii"), v) for k, v in obj.__dict__.items())

    fix_object(full_data)
    full_data.entity_occur_interval = {k.decode('utf8'): v for k, v in full_data.entity_occur_interval.items()}
    full_data.entity_sentiment_interval = {k.decode('utf8'): v for k, v in full_data.entity_sentiment_interval.items()}

    for day in full_data.days:
        fix_object(day)
        day.entity_sentiment_day = {k.decode('utf8'): v for k, v in day.entity_sentiment_day.items()}
        day.entity_occur_day = {k.decode('utf8'): v for k, v in day.entity_occur_day.items()}
        for news in day.day_news:
            fix_object(news)


def process_market_time_series(path, sheet, start_date=None, end_date=None):
    import xlrd

    def read_date(date):
        """
        In case conversion from the Excel-style datetime is needed - not currently in use
        :param date: Excel datetime object
        :return: Pandas-compliant datetime data type
        """
        return xlrd.xldate.xldate_as_datetime(date, 0)

    raw = pd.read_excel(path, sheetname=sheet)
    start_date = start_date if start_date and start_date >= raw.index[1] else raw.index[1]
    end_date = end_date if end_date and end_date <= raw.index[-1] else raw.index[-1]
    raw = raw[raw.index >= start_date]
    data = raw[raw.index <= end_date].astype('float64')
    # Truncate the price time series to be consistent with the count and sentiment time series
    daily_log_return = data.apply(lambda x: np.log(x) - np.log(x.shift(1)))
    daily_vol = daily_log_return.rolling(7).std() #Weekly volatility

    return data, daily_vol


def process_count_sentiment(FullData_obj, start_date=None, end_date=None, mode='weekly', include_list=None):
    """
    Process the FullData object to obtain corresponding count and sentiment time series
    :param FullData_obj: The FullData object containing all the unprocessed count and sentiment information.
    :param start_date: datetime, optional - start time of data. To be a valid argument, the start_date
    must be later than the start date of the FullData object
    :param end_date: datetime, optional - end time of the data. To be a valid argument, the end_date must be earlier
    than the end date of the FullData object
    :param period: the period for aggregation. E.g. 5 for weekly, 21 for monthly, 252 for yearly (trading day only)
    :param include_list: optional [] or {} - only entities in the list will be considered. If blank all entities
    will be considered
    :return:
    """

    def isEndofPeriod(day, mode):
        from pandas.tseries.offsets import MonthEnd
        if mode == 'monthly': return True if day == day + MonthEnd(0) else False
        else: return True if day.weekday() == 4 else False

    if start_date:
        start_date = start_date if start_date > FullData_obj.start_date else FullData_obj.start_date
    else:
        start_date = FullData_obj.start_date

    if end_date:
        end_date = end_date if end_date < FullData_obj.end_date else FullData_obj.end_date
    else:
        end_date = FullData_obj.end_date

    if include_list:
        if isinstance(include_list, dict):
            include_list = include_list.keys()
        elif isinstance(include_list, list):
            pass
        else:
            raise TypeError("include_list needs to be an iterable of type either list or dictionary")
    else:
        include_list = FullData_obj.entity_occur_interval.keys()

    # Initialise the aggregate list and counter
    aggregate_count_sum = []
    aggregate_sentiment_avg = []
    aggregate_sentiment_sum = []
    date_time_series = []

    # Initialise the weekly counters
    period_count_sum = Counter({})
    period_sentiment_sum = Counter({})

    i = 0
    for day in FullData_obj.days:
        if day.date < start_date:
            continue
        elif day.date == start_date:
            i = 0
        elif day.date > end_date:
            break

        if isEndofPeriod(day.date, mode):
            aggregate_count_sum.append(period_count_sum)
            period_sentiment_avg = {k: v / period_count_sum[k] if period_count_sum[k] != 0 else 1
                                        for k, v in period_sentiment_sum.items()}
            aggregate_sentiment_avg.append(period_sentiment_avg)
            aggregate_sentiment_sum.append(period_sentiment_sum)
            # Re-initialise the periodical sub-totals for the next period
            date_time_series.append(day.date)
            period_count_sum = Counter(dict((name, 0) for name in include_list))
            period_sentiment_sum = Counter(dict((name, 0) for name in include_list))
        else:
            for entity in include_list:
                if entity in day.entity_occur_day.keys():
                    period_count_sum[entity] += day.entity_occur_day[entity]
                    period_sentiment_sum[entity] += day.entity_sentiment_day[entity] * day.entity_occur_day[entity]
        i += 1

    date_time_series = pd.Series(date_time_series)
    count_time_series = pd.DataFrame(aggregate_count_sum).dropna(how='all')
    count_time_series["Time"] = date_time_series
    count_time_series.set_index('Time', inplace=True)
    sentiment_avg_time_series = pd.DataFrame(aggregate_sentiment_avg).dropna(how='all')
    sentiment_avg_time_series["Time"] = date_time_series
    sentiment_avg_time_series.set_index('Time', inplace=True)
    sentiment_sum_time_series = pd.DataFrame(aggregate_sentiment_sum).dropna(how='all')
    sentiment_sum_time_series["Time"] = date_time_series
    sentiment_sum_time_series.set_index('Time', inplace=True)

    return count_time_series, sentiment_avg_time_series, sentiment_sum_time_series


def get_top_n_entities(FullData_obj, n=200):
    """
    Return a dictionary of the entities that appear in the top-n ranks in the news covered in the FullData object.
    Key is the (abbreviated) names of the entities and Value is the number of times the entities appeared
    :param FullData_obj: a FullData Object
    :param n:
    :return:
    """
    return dict(sorted(FullData_obj.entity_occur_interval.items(), key=lambda x: x[1])[-n:])


def correlator(market_data, vol_data, sentiment_avg_data, sentiment_sum_data, count_data, plot=True):

    def normalised_corr(series1, series2, max_period=2):
        series1 = (series1 - np.mean(series1)) / (np.std(series1) * len(series1))
        series2 = (series2 - np.mean(series2)) / np.std(series2)
        correl = np.correlate(series1, series2, mode='full')
        res = correl[int((len(correl))/2):]
        abs_series = np.abs(res[:max_period+1])
        max_idx = np.argmax(abs_series)
        max_val = res[max_idx]
        return res, max_idx, max_val

    assert set(sentiment_avg_data.index) == set(count_data.index)
    intersect_dates = market_data.index.intersection(count_data.index)
    market_data = market_data.loc[intersect_dates].astype('float64')
    benchmark_data = (np.log(market_data['FED']) - np.log(market_data['FED'].shift(1)))[1:] # SPX 500 log-return

    market_data_log = market_data.apply(lambda x: np.log(x) - np.log(x.shift(1)))[1:].subtract(benchmark_data, axis='index')
    # Superior log return over SPX return
    vol = vol_data.loc[intersect_dates].astype('float64')

    # market_data_bin = market_data.apply(lambda x: np.sign(x-x.shift(1)))[1:]
    # Compute the log-return of the stock prices over the previous period. The first data is removed because it is a NaN
    res = []

    for single_name in list(market_data.columns):
        if "." in single_name:
            # Special cases for BoJ and Fed where there are two underlying securities...
            sentiment_series_avg = sentiment_avg_data[single_name.split(".")[0]].iloc[1:].astype('float64').values
            sentiment_series_sum = sentiment_sum_data[single_name.split(".")[0]].iloc[1:].astype('float64').values
            count_series = count_data[single_name.split(".")[0]].iloc[1:].astype('float64').values
            vol_series = vol[single_name].iloc[1:].astype('float64').values
            market_series = market_data_log[single_name].astype('float64').values
        else:
            sentiment_series_avg = sentiment_avg_data[single_name].iloc[1:].astype('float64').values
            sentiment_series_sum = sentiment_sum_data[single_name].iloc[1:].astype('float64').values
            count_series = count_data[single_name].iloc[1:].astype('float64').values
            vol_series = vol[single_name].iloc[1:].astype('float64').values
            # Log transform reduces the length of the series by one so truncate the first data point for sentiment and
            # count series as well.
            market_series = market_data_log[single_name].astype('float64').values

        non_nan_idx = ~np.isnan(market_series)

        market_series = market_series[non_nan_idx]
        count_series = count_series[non_nan_idx]
        sentiment_series_avg = sentiment_series_avg[non_nan_idx]
        sentiment_series_sum = sentiment_series_sum[non_nan_idx]
        vol_series = vol_series[non_nan_idx]
        # Tackle the missing data problem...

        corr_count_price, max_corr_count_price, max_corr_count_price_idx = normalised_corr(count_series, market_series)
        corr_avg_sentiment_price, max_avg_corr_sentiment_price, max_corr_avg_sentiment_price_idx = normalised_corr(sentiment_series_avg,
                                                                                               market_series)
        corr_sum_sentiment_price, max_sum_corr_sentiment_price, max_corr_sum_sentiment_price_idx = normalised_corr(
            sentiment_series_sum,
            market_series)
        corr_count_vol, max_corr_count_vol, max_corr_count_vol_idx = normalised_corr(count_series, vol_series)
        corr_vol_sum_sentiment, max_corr_vol_sum_sentiment, max_corr_vol_sum_sentiment_idx = normalised_corr(sentiment_series_sum, vol_series)
        corr_vol_price, max_corr_vol_price, max_corr_vol_price_idx = normalised_corr(market_series, vol_series)

        this_res = {'Name': single_name,
                    'PriceCountCorr': max_corr_count_price,
                    'PriceCountLag': max_corr_count_price_idx,
                    'PriceAvgSentimentCorr': max_avg_corr_sentiment_price,
                    'PriceAvgSentimentLag': max_corr_avg_sentiment_price_idx,
                    'PriceSumSentimentCorr': max_sum_corr_sentiment_price,
                    'PriceSumSentimentLag': max_corr_sum_sentiment_price_idx,
                    'VolCountCorr': max_corr_count_vol,
                    'VolCountLag': max_corr_count_vol_idx,
                    'VolSumSentimentCorr': max_corr_vol_sum_sentiment,
                    'VolSumSentimentLag': max_corr_vol_sum_sentiment_idx,
                    'SpotVolCorr': max_corr_vol_price,
                    'SpotVolLag': max_corr_vol_price_idx,
                    }
        print(this_res)
        res.append(this_res)

        if plot:
            # Perform the cross-correlation
            plt.subplot(221)
            plt.title('Count-Price Correlation')
            plt.plot(corr_count_price)
            plt.subplot(222)
            plt.title('Sentiment-Price Correlation')
            plt.plot(corr_avg_sentiment_price)
            plt.subplot(223)
            plt.title('Count-Vol Correlation')
            plt.plot(corr_count_vol)
            plt.subplot(224)
            plt.title('Spot-Vol Correlation')
            plt.plot(corr_vol_price)
            plt.show()

    pd.DataFrame(res).to_csv('CorrRes.csv')
    print('Done')

if __name__ == "__main__":

    full_data_obj = 'full.date.20061020-20131120'
    market_data_path = 'data/top_entities.xlsx'
    market_data_sheet = 'Sheet3'

    with open(full_data_obj, 'rb') as f:
        all_data = pickle.load(f, fix_imports=True, encoding='bytes')
    fix_fulldata(all_data)

    start_date = all_data.start_date
    end_date = all_data.end_date

    price_time_series, vol_time_series = process_market_time_series(market_data_path, market_data_sheet,
                                                    start_date=pd.to_datetime(start_date),
                                                    end_date=pd.to_datetime(end_date))

    count_time_series, sentiment_avg_time_series, sentiment_sum_time_series = process_count_sentiment(all_data,
                                                                    include_list=list(price_time_series.columns))
    correlator(price_time_series, vol_time_series, sentiment_avg_time_series, sentiment_sum_time_series, count_time_series, plot=False)