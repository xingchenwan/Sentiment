import networkx as nx

from source.utils import *
from source.all_classes import *
import statistics
import matplotlib.pylab as plt
from collections import Counter
import pandas as pd


def stat_tests(data, threshold=0.05):
    import scipy.stats as st
    if not isinstance(data, pd.Series) and not isinstance(data, np.ndarray):
        raise TypeError('The input data must be a pandas series or a numpy array')
    elif isinstance(data, pd.Series):
        data = data.values

    data = data[~np.isnan(data)]
    # Clear NaNs from the data-set
    # print(data)
    _, pval = st.mstats.normaltest(data)
    if pval > threshold:  # Passed normal distribution test
        print('Normal test passed - using t-tests')
    t, prob = st.ttest_1samp(data, 0)
    # Return the t-statistics and p-value on whether the data mean is significantly different from 0
    mean_5, mean_95 = st.t.interval(1-threshold, len(data)-1, loc=np.mean(data), scale=st.sem(data))
    print('Normal test statistics', pval,
          'T-statistic', t,
          '2-tailed p-value', prob,
          '95th Percentile', mean_95,
          '5th Percentile', mean_5)
    return t, prob, mean_5, mean_95


def fix_fulldata(full_data):
    """
    A temporary patch to resolve the incompatibility issue between Python 2.x and 3.x pickling
    :param full_data:
    :return:
    """

    def _fix_object(obj):
        """
        Decode the byte data type back to strings due to Python 3.x un-pickling
        :param obj: all_data object
        :return: None
        """
        obj.__dict__ = dict((k.decode("utf8"), v) for k, v in obj.__dict__.items())

    def _fix_iterables(obj):
        """
        Decode iterables
        :param obj: An iterable, dict or list (other iterables are not included as they were not included in the
        original data type design
        :return: Fixed iterables
        """
        if isinstance(obj, list):
            return [x.decode('utf8') if isinstance(x, bytes) else x for x in obj]
        elif isinstance(obj, dict):
            return {k.decode('utf8'): v for k, v in obj.items()}

    _fix_object(full_data)
    full_data.entity_occur_interval = _fix_iterables(full_data.entity_occur_interval)
    full_data.entity_sentiment_interval = _fix_iterables(full_data.entity_sentiment_interval)

    for day in full_data.days:
        _fix_object(day)
        day.entity_sentiment_day = _fix_iterables(day.entity_sentiment_day)
        day.entity_occur_day = _fix_iterables(day.entity_occur_day)
        for news in day.day_news:
            _fix_object(news)
            news.entity_occur = _fix_iterables(news.entity_occur)
            news.entity_sentiment = _fix_iterables(news.entity_sentiment)


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
    daily_vol = daily_log_return.rolling(7).std() # Weekly volatility

    return data, daily_vol


def process_count_sentiment(FullData_obj, start_date=None, end_date=None, mode='weekly', include_list=None):
    """
    Process the FullData object to obtain corresponding count and sentiment time series
    :param FullData_obj: The FullData object containing all the unprocessed count and sentiment information.
    :param start_date: datetime, optional - start time of data. To be a valid argument, the start_date
    must be later than the start date of the FullData object
    :param end_date: datetime, optional - end time of the data. To be a valid argument, the end_date must be earlier
    than the end date of the FullData object
    :param mode: the period for aggregation. 'weekly','monthly', 'daily'
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
    aggregate_sentiment_med = []
    aggregate_sentiment_sum = []
    date_time_series = []

    # Initialise the weekly counters
    period_count_sum = Counter({})
    period_sentiment_sum = Counter({})
    period_sentiment_med = Counter({})

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
            aggregate_sentiment_med.append({k: statistics.median(v) if len(v) else 0 for k, v in period_sentiment_med.items()})
            aggregate_sentiment_sum.append(period_sentiment_sum)
            # Re-initialise the periodical sub-totals for the next period
            date_time_series.append(day.date)
            period_count_sum = Counter(dict((name, 0) for name in include_list))
            period_sentiment_med = Counter(dict((name, []) for name in include_list))
            period_sentiment_sum = Counter(dict((name, 0) for name in include_list))

        else:
            for news in day.day_news:
                for entity in include_list:
                    if entity in news.entity_occur.keys():
                        period_count_sum[entity] += news.entity_occur[entity]
                        period_sentiment_med[entity].append(news.entity_sentiment[entity])
                        period_sentiment_sum[entity] += news.entity_sentiment[entity]*news.entity_occur[entity]
        i += 1

    date_time_series = pd.Series(date_time_series)
    count_time_series = pd.DataFrame(aggregate_count_sum).dropna(how='all')
    count_time_series["Time"] = date_time_series
    count_time_series.set_index('Time', inplace=True)
    sentiment_med_time_series = pd.DataFrame(aggregate_sentiment_med).dropna(how='all')
    sentiment_med_time_series["Time"] = date_time_series
    sentiment_med_time_series.set_index('Time', inplace=True)
    sentiment_sum_time_series = pd.DataFrame(aggregate_sentiment_sum).dropna(how='all')
    sentiment_sum_time_series["Time"] = date_time_series
    sentiment_sum_time_series.set_index('Time', inplace=True)

    return count_time_series, sentiment_med_time_series, sentiment_sum_time_series


def get_top_n_entities(FullData_obj, n=200):
    """
    Return a dictionary of the entities that appear in the top-n ranks in the news covered in the FullData object.
    Key is the (abbreviated) names of the entities and Value is the number of times the entities appeared
    :param FullData_obj: a FullData Object
    :param n:
    :return:
    """
    return dict(sorted(FullData_obj.entity_occur_interval.items(), key=lambda x: x[1])[-n:])


def correlator(market_data, vol_data, sentiment_med_data, sentiment_sum_data, count_data, plot=True, save_csv=False,
               display_summary_stats=True):

    def normalised_corr(series1, series2, max_period=3):
        """
        Compute the normalised cross-correlation between series1 and series2
        :param series1: Series1, Pandas series or numpy array
        :param series2: Series2, Pandas series or numpy array
        :param max_period: Maximum lag between series1 and series2 allowed
        :return: res: normalised cross-correlation coefficient, max_val: maximum correlation coefficient within the
        allowed period of lag in terms of magnitude, max_idx: the lag (index) corresponding to that correlation
        coefficient
        """
        series1 = (series1 - np.mean(series1)) / (np.std(series1) * len(series1))
        series2 = (series2 - np.mean(series2)) / np.std(series2)
        correl = np.correlate(series1, series2, mode='full')
        res = correl[int((len(correl))/2):]
        abs_series = np.abs(res[:max_period+1])
        max_idx = np.argmax(abs_series)
        max_val = res[max_idx]
        return res, max_val, max_idx

    assert set(sentiment_med_data.index) == set(count_data.index)
    intersect_dates = market_data.index.intersection(count_data.index)
    market_data = market_data.loc[intersect_dates].astype('float64')
    benchmark_data = (np.log(market_data['_ALL']) - np.log(market_data['_ALL'].shift(1)))[1:]  # MSCI log-return

    market_data_log = market_data.apply(lambda x: np.log(x) - np.log(x.shift(1)))[1:].subtract(benchmark_data, axis='index')
    # Superior log return over SPX return
    vol = vol_data.loc[intersect_dates].astype('float64')
    # market_data_bin = market_data.apply(lambda x: np.sign(x-x.shift(1)))[1:]

    # Compute the log-return of the stock prices over the previous period. The first data is removed because it is a NaN
    res = []

    for single_name in list(market_data.columns):
        if "." in single_name:
            # Special cases for BoJ and Fed where there are two underlying securities...
            sentiment_series_med = sentiment_med_data[single_name.split(".")[0]].iloc[1:].astype('float64').values
            sentiment_series_sum = sentiment_sum_data[single_name.split(".")[0]].iloc[1:].astype('float64').values
            count_series = count_data[single_name.split(".")[0]].iloc[1:].astype('float64').values
            vol_series = vol[single_name].iloc[1:].astype('float64').values
            market_series = market_data_log[single_name].astype('float64').values
        else:
            sentiment_series_med = sentiment_med_data[single_name].iloc[1:].astype('float64').values
            sentiment_series_sum = sentiment_sum_data[single_name].iloc[1:].astype('float64').values
            count_series = count_data[single_name].iloc[1:].astype('float64').values
            vol_series = vol[single_name].iloc[1:].astype('float64').values
            # Log transform reduces the length of the series by one so truncate the first data point for sentiment and
            # count series as well.
            market_series = market_data_log[single_name].astype('float64').values

        non_nan_idx = ~np.isnan(market_series)

        market_series = market_series[non_nan_idx]
        count_series = count_series[non_nan_idx]
        sentiment_series_med = sentiment_series_med[non_nan_idx]
        sentiment_series_sum = sentiment_series_sum[non_nan_idx]
        vol_series = vol_series[non_nan_idx]
        # Tackle the missing data problem...

        corr_count_price, max_corr_count_price, max_corr_count_price_idx = normalised_corr(count_series, market_series)

        corr_med_sentiment_price, max_med_corr_sentiment_price, \
            max_corr_med_sentiment_price_idx = normalised_corr(sentiment_series_med, market_series)

        corr_sum_sentiment_price, max_sum_corr_sentiment_price, \
            max_corr_sum_sentiment_price_idx = normalised_corr(sentiment_series_sum, market_series)

        corr_count_vol, max_corr_count_vol, max_corr_count_vol_idx = normalised_corr(count_series, vol_series)

        corr_vol_sum_sentiment, max_corr_vol_sum_sentiment, \
            max_corr_vol_sum_sentiment_idx = normalised_corr(sentiment_series_sum, vol_series)

        corr_vol_price, max_corr_vol_price, max_corr_vol_price_idx = normalised_corr(market_series, vol_series)

        this_res = {'Name': single_name,
                    'PriceCountCorr': max_corr_count_price,
                    'PriceCountLag': max_corr_count_price_idx,
                    'PriceMedSentimentCorr': max_med_corr_sentiment_price,
                    'PriceMedSentimentLag': max_corr_med_sentiment_price_idx,
                    'PriceSumSentimentCorr': max_sum_corr_sentiment_price,
                    'PriceSumSentimentLag': max_corr_sum_sentiment_price_idx,
                    'VolCountCorr': max_corr_count_vol,
                    'VolCountLag': max_corr_count_vol_idx,
                    'VolMedSentimentCorr': max_corr_vol_sum_sentiment,
                    'VolMedSentimentLag': max_corr_vol_sum_sentiment_idx,
                    'SpotVolCorr': max_corr_vol_price,
                    'SpotVolLag': max_corr_vol_price_idx,
                    }
        print(this_res)
        res.append(this_res)

        if plot:
            # Perform the cross-correlation
            plt.subplot(231)
            plt.title('Count-Price Correlation')
            plt.plot(corr_count_price)
            plt.subplot(232)
            plt.title('MedSentiment-Price Correlation')
            plt.plot(corr_med_sentiment_price)
            plt.subplot(233)
            plt.title('SumSentiment-Price Correlation')
            plt.plot(corr_sum_sentiment_price)
            plt.subplot(234)
            plt.title('Count-Vol Correlation')
            plt.plot(corr_count_vol)
            plt.subplot(235)
            plt.title('Spot-Vol Correlation')
            plt.plot(corr_vol_price)
            plt.show()

    res = pd.DataFrame(res)

    if display_summary_stats: print(res.describe())

    if save_csv:
        pd.DataFrame(res).to_csv('CorrRes.csv')
        print('Done')
    return res


def plot_scatter(data, data_labels, x_label=None, y_label=None, categories=None):
    assert len(data) == len(data_labels)
    if categories:
        assert isinstance(categories, dict)
        assert len(categories) == len(data)
        color_map_keys = list(set(categories.values()))
        color_map_values = [i for i in range(len(color_map_keys))]
        color_map_dict = dict(zip(color_map_keys, color_map_values))
        color_map = np.array([color_map_dict[categories[name]] for name in data_labels])
    else:
        color_map = np.ones((len(data), 1))

    axis = np.linspace(0, 100, len(data))
    fig, ax = plt.subplots()
    ax.scatter(axis, data, c=color_map, s=10)
    plt.axhline(0, color="gray")
    plt.axhline(np.nanmean(data), color="red")
    plt.axhline(np.nanmedian(data), color="blue")
    if x_label: plt.xlabel(x_label)
    if y_label: plt.ylabel(y_label)

    if len(data_labels):
        for i in range(len(data_labels)):
            ax.annotate(data_labels.iloc[i], (axis[i], data[i]), fontsize=5)


def get_sector(path, name_col, sector_col, names):
    """
    Get the ground truth sectors dictionaries of a series of single names
    :param path:
    :param name_col:
    :param sector_col:
    :return:
    """
    data = pd.read_csv(path, header=None)
    if isinstance(name_col, str):
        assert name_col in list(data.columns)
        name_column = data[name_col]
    elif isinstance(name_col, int):
        assert name_col <= len(data.columns)
        name_column = data.iloc[:, name_col]
    else:
        raise TypeError("Invalid name column data type")

    if isinstance(sector_col, str):
        assert sector_col in list(data.columns)
        sector_column = data[sector_col]
    elif isinstance(sector_col, int):
        assert sector_col <= len(data.columns)
        sector_column = data.iloc[:, sector_col]
    else:
        raise TypeError("Invalid sector column data type")

    sector_lookup = pd.Series(sector_column.values, index=name_column).to_dict()
    # print("Names not found in the lookup:", [i for i in names if i not in sector_lookup.keys()])
    return {name: sector_lookup[name] if name in sector_lookup.keys() else 'Unclassified' for name in names}


if __name__ == "__main__":

    full_data_obj = 'full.date.20061020-20131120'
    market_data_path = 'data/top_entities.xlsx'
    market_data_sheet = 'Sheet3'
    sector_path = "source/sector.csv"

    with open(full_data_obj, 'rb') as f:
        all_data = pickle.load(f, fix_imports=True, encoding='bytes')
    fix_fulldata(all_data)

    start_date = all_data.start_date
    end_date = all_data.end_date

    price_time_series, vol_time_series = process_market_time_series(market_data_path, market_data_sheet,
                                                                    start_date=pd.to_datetime(start_date),
                                                                    end_date=pd.to_datetime(end_date))

    # nx_graph = all_data.build_occurrence_network_graph(focus_dict=list(price_time_series.columns))

    count_time_series, sentiment_avg_time_series, sentiment_sum_time_series = process_count_sentiment(all_data,
                                                                    include_list=list(price_time_series.columns))
    res = correlator(price_time_series, vol_time_series, sentiment_avg_time_series, sentiment_sum_time_series,
                     count_time_series, plot=False, save_csv=True, display_summary_stats=False)

    #stat_tests(res['PriceCountCorr'])
    #stat_tests(res['PriceMedSentimentCorr'])
    #stat_tests(res['PriceSumSentimentCorr'])
    #stat_tests(res['VolCountCorr'])
    #stat_tests(res['SpotVolCorr'])

    sector = (get_sector(sector_path, 0, 2, res['Name']))
    plot_scatter(res['PriceMedSentimentCorr'], res['Name'],
                 x_label='Stock Names', y_label='Correlation Coefficient', categories=sector)

    plt.show()
