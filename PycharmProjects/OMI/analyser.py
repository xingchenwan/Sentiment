# Analyser: modules that perform numerical analyses on the result
# Xingchen Wan | Xingchen.Wan@st-annes.ox.ac.uk | Oxford-Man Institute of Quantitative Finance

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pylab as plt


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
    correl = np.correlate(series2, series1, mode='full')
    res = correl[int((len(correl)) / 2):]
    #res = correl[:]
    # print(len(res))
    if fix_lag is None:
        abs_series = np.abs(res[:max_lag + 1])
        max_idx = np.argmax(abs_series)
        max_val = res[max_idx]
    else:
        max_idx = fix_lag
        max_val = res[fix_lag - 1]
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
        include_names = [x for x in focus_iterable if x in market_data.columns]
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


def rolling_x_correlate(roll_window_size, frame1, frame2, start_date=None, end_date=None,
                        fix_lag=None, max_lag=2, plot=True):
    data = pd.concat([frame1, frame2], axis=1, join='inner')
    start_date = start_date if start_date and start_date > data.index[0] else \
        data.index[0]
    end_date = end_date if end_date and end_date < data.index[-1] else \
        data.index[-1]
    rolling_window_start = start_date + pd.to_timedelta(roll_window_size, unit='d')
    res = pd.Series(0, index=data.index[data.index >= rolling_window_start])
    res = res[res.index <= end_date]
    i = 0
    for dt in res.index:
        sub_data_1 = data.iloc[:, 0]
        sub_data_1 = sub_data_1[dt-pd.to_timedelta(roll_window_size, unit='d'):dt].values
        sub_data_2 = data.iloc[:, 1]
        sub_data_2 = sub_data_2[dt-pd.to_timedelta(roll_window_size, unit='d'):dt].values
        if fix_lag is not None:
            _, res.iloc[i], _ = normalised_corr(sub_data_1, sub_data_2, fix_lag=fix_lag)
        else:
            _, res.iloc[i], _ = normalised_corr(sub_data_1, sub_data_2, max_lag=max_lag)
        i += 1
    if plot:
        res.plot()
    return res