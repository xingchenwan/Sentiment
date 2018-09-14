# Analyser: modules that perform numerical analyses on the result
# Xingchen Wan | Xingchen.Wan@st-annes.ox.ac.uk | Oxford-Man Institute of Quantitative Finance

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pylab as plt
from pandas.tseries.offsets import BDay


def normalised_corr(series1, series2, max_lag=2, fix_lag=None):
    """
    Compute the normalised cross-correlation between series1 and series2
    :param series1: Series1, Pandas series or numpy array
    :param series2: Series2, Pandas series or numpy array
    :param max_lag: Maximum lag between series1 and series2 allowed
    :param fix_lag: fixed lag - this argument overrides the max_lag argument.
    :return: res: normalised cross-correlation coefficients, max_val: maximum correlation coefficient within the
    allowed period of lag in terms of magnitude, max_idx: the lag (index) corresponding to that correlation
    coefficient
    """
    if isinstance(series1, pd.Series): series1 = series1.values
    if isinstance(series2, pd.Series): series2 = series2.values
    series1 = (series1 - np.mean(series1)) / (np.std(series1) * len(series1))
    series2 = (series2 - np.mean(series2)) / np.std(series2)
    correl = np.correlate(series1, series2, mode='full')
    res = correl[int((len(correl)) / 2):]
    #res = correl[:]
    # print(len(res))
    if fix_lag is None:
        abs_series = np.abs(res[:max_lag + 1])
        max_idx = np.argmax(abs_series)
        max_val = res[max_idx]
    else:
        max_idx = fix_lag
        max_val = res[fix_lag]
    return res, max_val, max_idx


def correlate(market_data, vol_data, sentiment_med_data, sentiment_sum_data, count_data, max_lag=2,
              count_threshold=0,
              plot=True, save_csv=False, fix_lag=None,
              display_summary_stats=True,
              focus_iterable=None):

    assert set(sentiment_med_data.index) == set(count_data.index)
    assert set(sentiment_sum_data.index) == set(count_data.index)
    intersect_dates = market_data.index.intersection(count_data.index)
    market_data = market_data.loc[intersect_dates].astype('float64')
    benchmark_data = market_data['_ALL'] # MSCI log-return

    market_data = market_data.subtract(benchmark_data, axis='index')
    # Superior log return over SPX return
    vol = vol_data.loc[intersect_dates].astype('float64')
    # market_data_bin = market_data.apply(lambda x: np.sign(x-x.shift(1)))[1:]

    res = []
    if focus_iterable:
        include_names = [x for x in focus_iterable if x in list(market_data.columns)]
    else:
        include_names = list(market_data.columns)

    for single_name in include_names:
        if "." in single_name:
            # Special cases for BoJ and Fed where there are two underlying securities...
            sentiment_series_med = sentiment_med_data[single_name.split(".")[0]].astype('float64').values
            sentiment_series_sum = sentiment_sum_data[single_name.split(".")[0]].astype('float64').values
            count_series = count_data[single_name.split(".")[0]].astype('float64').values
            vol_series = vol[single_name].astype('float64').values
            market_series = market_data[single_name].astype('float64').values
        else:
            sentiment_series_med = sentiment_med_data[single_name].astype('float64').values
            sentiment_series_sum = sentiment_sum_data[single_name].astype('float64').values
            count_series = count_data[single_name].astype('float64').values
            vol_series = vol[single_name].astype('float64').values
            market_series = market_data[single_name].astype('float64').values

        non_nan_idx = ~np.isnan(market_series)
        if count_threshold > 0:
            non_nan_idx = np.logical_and(~np.isnan(market_series), (count_series > count_threshold))
        all_nan = True if (non_nan_idx == False).all() else False
        # Flag that the entire numpy series consists of NaN

        if not all_nan:
            market_series = market_series[non_nan_idx]
            count_series = count_series[non_nan_idx]
            sentiment_series_med = sentiment_series_med[non_nan_idx]
            sentiment_series_sum = sentiment_series_sum[non_nan_idx]
            vol_series = vol_series[non_nan_idx]
            # Then prune the data to get rid of the NaNs
            all_series = [market_series, count_series, sentiment_series_med, sentiment_series_sum, vol_series]


            corr_count_price, max_corr_count_price, max_corr_count_price_idx = normalised_corr(count_series, market_series, max_lag=max_lag, fix_lag=fix_lag)

            corr_med_sentiment_price, max_med_corr_sentiment_price, \
                max_corr_med_sentiment_price_idx = normalised_corr(sentiment_series_med, market_series, max_lag=max_lag, fix_lag=fix_lag)

            corr_sum_sentiment_price, max_sum_corr_sentiment_price, \
                max_corr_sum_sentiment_price_idx = normalised_corr(sentiment_series_sum, market_series, max_lag=max_lag, fix_lag=fix_lag)

            corr_count_vol, max_corr_count_vol, max_corr_count_vol_idx = normalised_corr(count_series, vol_series, max_lag=max_lag, fix_lag=fix_lag)

            corr_vol_sum_sentiment, max_corr_vol_sum_sentiment, \
                max_corr_vol_sum_sentiment_idx = normalised_corr(sentiment_series_sum, vol_series, max_lag=max_lag, fix_lag=fix_lag)

            corr_vol_price, max_corr_vol_price, max_corr_vol_price_idx = normalised_corr(market_series, vol_series, max_lag=max_lag, fix_lag=fix_lag)

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
        else:
            this_res = {'Name': single_name,
                        'PriceCountCorr': np.nan,
                        'PriceCountLag': np.nan,
                        'PriceMedSentimentCorr': np.nan,
                        'PriceMedSentimentLag': np.nan,
                        'PriceSumSentimentCorr': np.nan,
                        'PriceSumSentimentLag': np.nan,
                        'VolCountCorr': np.nan,
                        'VolCountLag': np.nan,
                        'VolMedSentimentCorr': np.nan,
                        'VolMedSentimentLag': np.nan,
                        'SpotVolCorr': np.nan,
                        'SpotVolLag': np.nan,
                        }
            print('Empty series occurred at ', single_name)
        res.append(this_res)

        if plot and not all_nan:
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


def causality(series1, series2, maxlag=2):
    from statsmodels.tsa.stattools import grangercausalitytests
    combined = pd.concat([series1.astype('float64'), series2.astype('float64')], axis=1, join='inner')
    combined = combined.values
    combined = combined[~np.isnan(combined).any(axis=1)]
    print(grangercausalitytests(combined, maxlag=maxlag))


def stat_tests(data, threshold=0.05):
    """
    Perform a series of statistic tests: normal distribution test, t-test and confidence interval
    :param data: array-like
    :param threshold: threshold to reject null hypothesis. Default value is 0.05
    :return:
    """
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


def rolling_correlate(name, roll_window_size, *input_frames, frame_names=[], fix_lag=None, max_lag=2,
                      start_date=None, end_date=None, exclude_pairs=None, include_pairs=None,
                      count_threshold={}):

    if include_pairs:
        if not isinstance(include_pairs, list):
            raise TypeError("include_pairs need to be list of tuples with length 2")
        elif not isinstance(include_pairs[0], tuple):
            raise TypeError("include_pairs need to be list of tuples with length 2")
        elif len(include_pairs[0]) != 2:
            raise TypeError("include_pairs need to be list of tuples with length 2")

    if exclude_pairs:
        if not isinstance(exclude_pairs, list):
            raise TypeError("Exclude_pairs need to be list of tuples with length 2")
        elif not isinstance(exclude_pairs[0], tuple):
            raise TypeError("Exclude_pairs need to be list of tuples with length 2")
        elif len(exclude_pairs[0]) != 2:
            raise TypeError("Exclude_pairs need to be list of tuples with length 2")
    if frame_names and len(frame_names) != len(input_frames):
        raise ValueError("Mismatch between data length and data_names length.")

    series_collection = []
    i = 0
    for frame in input_frames:
        try:
            series = frame[name].astype('float64')
            series.name = frame_names[i]
            series_collection.append(series)
        except KeyError:
            print(name, " is not found in data argument position ", input_frames.index(frame))
        i += 1
    data = pd.concat(series_collection, axis=1, join='inner')
    # data only contains where the dates of the different data frames intersect
    if start_date:
        start_date = pd.to_datetime(start_date)
        if start_date > data.index[0]:
            data = data[data.index >= start_date]
        else: start_date = data.index[0]
    else: start_date = data.index[0]

    if end_date:
        end_date = pd.to_datetime(end_date)
        if end_date < data.index[-1]:
            data = data[data.index <= end_date]
        else: end_date = data.index[-1]
    else: end_date = data.index[-1]
    # Truncate the the data frame in accordance to the specified start and end dates
    res = {}

    roll_window_start = start_date+pd.to_timedelta(roll_window_size, unit='d')

    for i in range(len(data.columns)):
        for j in range(i, len(data.columns)):
            col_name1 = data.columns[i]
            col_name2 = data.columns[j]
            if exclude_pairs and ((col_name1, col_name2) in exclude_pairs or (col_name2, col_name1) in exclude_pairs):
                continue
            if include_pairs and ((col_name1, col_name2) not in include_pairs and (col_name2, col_name1) not in include_pairs):
                continue
            local_data1 = data[col_name1]
            local_data2 = data[col_name2]
            this_res = pd.Series(0, index=data.index[data.index >= roll_window_start])
            this_res.name = "Corr" + col_name1 + "To" + col_name2

            k = 0
            for dt in this_res.index:
                series1 = local_data1[dt-pd.to_timedelta(roll_window_size, unit='d'):dt].values
                series2 = local_data2[dt-pd.to_timedelta(roll_window_size, unit='d'):dt].values
                if count_threshold:
                    if col_name1 in count_threshold.keys():
                        series1 = series1[series1 > count_threshold[col_name1]]
                        series2 = series2[series1 > count_threshold[col_name1]]
                    if col_name2 in count_threshold.keys():
                        series1 = series1[series2 > count_threshold[col_name2]]
                        series2 = series2[series2 > count_threshold[col_name2]]
                # Select appropriate date range
                if fix_lag is not None:
                    _, this_res.iloc[k], _ = normalised_corr(series1, series2, fix_lag=fix_lag)
                else:
                    _, this_res.iloc[k], _ = normalised_corr(series1, series2, max_lag=max_lag)
                k += 1
            res[this_res.name] = this_res
    return pd.DataFrame(res)

#######################################################
# Event-based analysis tools:


def _get_z_score(frame, window):
    r = frame.rolling(window=window)
    m = r.mean()
    s = r.std(ddof=0)
    return (frame-m)/s


def _get_event_series(z_score_series, threshold, ignore_window=10):
    event_series = pd.Series(0, index=z_score_series.index)
    ignore_cnt = 0
    for i in range(len(z_score_series)):
        if ignore_cnt != 0:
            ignore_cnt -= 1
        pt = z_score_series[i]
        if pt > threshold: status = 1
        elif pt < -threshold: status = -1
        else: status = 0

        if ignore_cnt: status = 0
        if status != 0 and ignore_cnt == 0:
            ignore_cnt = ignore_window
        event_series.iloc[i] = status
    return event_series


def vol_event_analyser(names, vol_frame, exogen_frame, rolling_window=90, detection_threshold=3, max_lag=5):

    def get_event_vol(name, event_series, vol_series, max_lag):
        event_cnt = event_series[event_series == 1].count()
        res = pd.DataFrame(0, index=[i for i in range(event_cnt)], columns=['Name', 'EventDate'] +
                                                                           [i for i in range(-2*max_lag, 1)])
        i = 0
        for day in range(len(event_series)):
            if event_series.iloc[day] == 1:
                dt = event_series.index[day]
                res.loc[i, 'Name'] = name
                res.loc[i, 'EventDate'] = dt
                for lag in range(-2*max_lag, 1):
                    res.loc[i, lag] = vol_series[dt+BDay(lag)]
        return res

    res = pd.DataFrame(columns=['Name', 'EventDate'] + [i for i in range(-2*max_lag, 1)])
    for name in names:
        exogen_series = exogen_frame[name]
        exogen_z_score = _get_z_score(exogen_series, rolling_window)
        event = _get_event_series(exogen_z_score, detection_threshold)
        this_res = get_event_vol(name, event, vol_frame[name], max_lag)
        res = res.append(this_res, ignore_index=True)
    print('Summary Stats:')
    print(res.describe())
    print('T-test')
    print(st.ttest_1samp(res[list(range(-2*max_lag, 1))].values, popmean=0)[1])
    res.to_csv('volEvent.csv')
    return res


def event_analyser(names, log_return_frame, exogen_frame, rolling_window=90, detection_threshold=3, max_lag=5):

    def get_abnormal_return(name, log_return_frame, date, window):
        # The normal return is given by the Market Model over MSCI return.
        open_date = date - pd.to_timedelta(window, 'D')
        hist_returns = log_return_frame[['_ALL', name]]
        hist_returns = hist_returns[(hist_returns.index >= open_date) & (hist_returns.index < date)]

        mkt_return_today = log_return_frame.loc[date, '_ALL']
        stock_return_today = log_return_frame.loc[date, name]

        # Estimate alpha and beta using simple OLS
        X = hist_returns['_ALL'].values
        y = hist_returns[name].values
        beta, alpha, r_val, p_val, std_err = st.linregress(X, y)
        # disturb_var = 1/(window - 2)*np.sum(np.square(y - alpha - beta*X))
        # print(beta)
        abnormal_return = stock_return_today - alpha - beta * mkt_return_today
        return abnormal_return

    def get_event_cumulative_abnormal_return(name, event_series, daily_abnormal_return_series, max_lag):
        event_cnt = event_series[event_series == 1].count() + event_series[event_series == -1].count()
        res = pd.DataFrame(0, index=[i for i in range(event_cnt)], columns=['Name','EventDate', 'Type'] + [i for i in range(-max_lag, max_lag+1)])
        i = 0
        for day in range(len(event_series)):
            if event_series.iloc[day] != 0:
                dt = event_series.index[day]
                res.loc[i, 'Name'] = name
                res.loc[i, 'EventDate'] = dt
                res.loc[i, 'Type'] = event_series.iloc[day]
                cum_rtn = 0
                for lag in range(-max_lag, max_lag+1):
                    cum_rtn += daily_abnormal_return_series[dt+BDay(lag)]
                    res.loc[i, lag] = cum_rtn
                i += 1
        return res

    res = pd.DataFrame(columns=['Name','EventDate', 'Type'] + [i for i in range(-max_lag, max_lag+1)])
    for name in names:
        daily_abnormal_return = pd.Series(np.nan, index=log_return_frame.index)
        for date in log_return_frame.index:
            if date < log_return_frame.index[0] + pd.to_timedelta(rolling_window, 'D'):
                continue
            daily_abnormal_return[date] = get_abnormal_return(name, log_return_frame, date, window=180)

        exogen_series = exogen_frame[name]
        exogen_z_score = _get_z_score(exogen_series, rolling_window)
        event = _get_event_series(exogen_z_score, detection_threshold)
        this_res = get_event_cumulative_abnormal_return(name, event, daily_abnormal_return, max_lag)
        res = res.append(this_res, ignore_index=True)
    pos_res = res[res['Type'] == 1]
    neg_res = res[res['Type'] == -1]

    print('Summary Stats:Pos/Neg')
    print(pos_res.describe())
    print(neg_res.describe())

    print('T-tests: Pos')
    print(st.ttest_1samp(pos_res[list(range(-max_lag, max_lag+1))].values, popmean=0)[1])
    print('T-tests: Neg')
    print(st.ttest_1samp(neg_res[list(range(-max_lag, max_lag+1))].values, popmean=0)[1])

    #pos_res.to_csv('posEvent.csv')
    #neg_res.to_csv('negEvent.csv')
    return pos_res, neg_res
