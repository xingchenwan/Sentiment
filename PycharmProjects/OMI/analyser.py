# Analyser: modules that perform numerical analyses on the result
# Xingchen Wan | Xingchen.Wan@st-annes.ox.ac.uk | Oxford-Man Institute of Quantitative Finance

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pylab as plt
from pandas.tseries.offsets import BDay
from processor import process_beta_time_series


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
    t, prob = st.ttest_1samp(data, 0)
    wilcox, prob2 = st.wilcoxon(data)
    # Return the t-statistics and p-value on whether the data mean is significantly different from 0
    mean_5, mean_95 = st.t.interval(1-threshold, len(data)-1, loc=np.mean(data), scale=st.sem(data))
    print(
          'T-statistic', t,
          'T-test p-value', prob,
          'WilCox', prob2,
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


def px_correlate(name1, *names, px_frame, correlation_period=90, start_date=None, end_date=None):
    if start_date is None: start_date = px_frame.index[0]
    if end_date is None: end_date = px_frame.index[-1]
    px_frame = px_frame[(px_frame.index >= start_date) & (px_frame.index <= end_date)]
    res = pd.DataFrame(np.nan, columns=names, index=px_frame.index)
    for dt in px_frame.index:
        if dt >= start_date + pd.to_timedelta(correlation_period):
            sub_frame = px_frame[(px_frame.index >= dt - pd.to_timedelta(correlation_period, 'D')) & (px_frame.index < dt)]
            px1 = sub_frame[name1]
            for name in names:
                px2 = sub_frame[name]
                res.loc[dt, name] = np.corrcoef(px1, px2)[0, 1]
    return res


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


def get_abnormal_return(name, log_return_frame, date, window, beta_path='data/^GSPC.csv'):
    # The normal return is given by the Market Model over MSCI return over the last window
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


def get_event_cumulative_abnormal_return(name, event_series, z_series, daily_abnormal_return_series, max_lag):
    event_cnt = event_series[event_series == 1].count() + event_series[event_series == -1].count()
    day_range = [i for i in range(-max_lag, max_lag + 1)]
    res = pd.DataFrame(0, index=[i for i in range(event_cnt)], columns=['Name','EventDate', 'Type'] + day_range)
    z_res = pd.DataFrame(0, index=res.index, columns=res.columns)
    i = 0
    for day in range(len(event_series)):
        if event_series.iloc[day] != 0:
            dt = event_series.index[day]
            res.loc[i, 'Name'] = name
            res.loc[i, 'EventDate'] = dt
            res.loc[i, 'Type'] = event_series.iloc[day]

            z_res.loc[i, 'Name'] = name
            z_res.loc[i, 'EventDate'] = dt
            z_res.loc[i, 'Type'] = event_series.iloc[day]

            cum_rtn = 0
            for lag in range(-max_lag, max_lag+1):
                try:
                    cum_rtn += daily_abnormal_return_series[dt+BDay(lag)]
                except KeyError:
                    pass
                res.loc[i, lag] = cum_rtn
                try:
                    z_res.loc[i, lag] = z_series[dt+BDay(lag)]
                except KeyError:
                    z_res.loc[i, lag] = np.nan
            i += 1
    return res, z_res


def get_event_vol(name, event_series, z_series, vol_series, max_lag):
    event_cnt = event_series[event_series == 1].count() + event_series[event_series == -1].count()
    day_range = [i for i in range(-max_lag, max_lag + 1)]
    res = pd.DataFrame(np.nan, index=[i for i in range(event_cnt)], columns=['Name', 'EventDate'] + day_range)
    z_res = pd.DataFrame(np.nan, index=res.index, columns=res.columns)
    i = 0
    for day in range(len(event_series)):
        if event_series.iloc[day] != 0:
            dt = event_series.index[day]
            res.loc[i, 'Name'] = name
            res.loc[i, 'EventDate'] = dt
            res.loc[i, 'Type'] = event_series.iloc[day]

            #z_res.loc[i, 'Name'] = name
            #z_res.loc[i, 'EventDate'] = dt
            #z_res.loc[i, 'Type'] = event_series.iloc[day]

            for lag in day_range:
                try:
                    res.loc[i, lag] = vol_series[dt+BDay(lag)] if vol_series[dt+BDay(lag)] != 0 else np.nan
                except KeyError:
                    res.loc[i, lag] = np.nan
            #    try:
             #       z_res.loc[i, lag] = z_series[dt+BDay(lag)]
            #    except KeyError:
            #        z_res.loc[i, lag] = np.nan
            i += 1
    return res, z_res


def get_avg_col(names, market_frame):
    # Get the average volatility of a single name / a collection of names (e.g. all companies within a sector
    res = pd.Series(np.nan, index=names)
    for name in names:
        # Retrieve all volatility
        res[name] = np.nanmean(market_frame[name])
    all_average = np.nanmean(res)
    print("Average Vol", all_average)
    return res, all_average


def event_analyser(names, market_frame, exogen_frame, rolling_window=180, detection_threshold=2.5, max_lag=5,
                   save_csv=True, mode='rtn', start_date=None, end_date=None):
    if mode == 'rtn':
        day_range = [i for i in range(-max_lag, max_lag+1)]
    elif mode == 'vol':
        day_range = [i for i in range(-max_lag, max_lag+1)]
    else:
        raise ValueError("Unrecognised mode argument: rtn or vol allowed")

    res = pd.DataFrame(columns=['Name','EventDate', 'Type'] + day_range)
    z_res = pd.DataFrame(columns=res.columns)

    start_date = pd.Timestamp(start_date) if start_date and pd.Timestamp(start_date) > market_frame.index[0] else market_frame.index[0]
    end_date = pd.Timestamp(end_date) if end_date and pd.Timestamp(end_date) < market_frame.index[-1] else market_frame.index[-1]
    market_frame = market_frame[(market_frame.index >= start_date) & (market_frame.index <= end_date)]
    exogen_frame = exogen_frame[(exogen_frame.index >= start_date) & (exogen_frame.index <= end_date)]

    for name in names:
        if mode == 'rtn':
            daily_abnormal_return = pd.Series(np.nan, index=market_frame.index)
            for date in market_frame.index:
                if date < start_date + pd.to_timedelta(rolling_window, 'D'):
                    continue
                elif date > end_date:
                    break
                daily_abnormal_return[date] = get_abnormal_return(name, market_frame, date, window=180)

        exogen_series = exogen_frame[name]
        exogen_z_score = _get_z_score(exogen_series, rolling_window)
        event = _get_event_series(exogen_z_score, detection_threshold)

        #pos_event_idx = event[event == 1].index
        #neg_event_idx = event[event == -1].index
        #plt.scatter(pos_event_idx, exogen_series[pos_event_idx], marker='^', color = 'r', label='Positive Events')
        #plt.scatter(neg_event_idx, exogen_series[neg_event_idx], marker='v', color='r', label='Negative Events')
        #exogen_series.plot()
        #plt.ylabel('SumSentiment')
        #plt.show()

        if mode == 'rtn':
            this_res, this_z_res = get_event_cumulative_abnormal_return(name, event, exogen_z_score, daily_abnormal_return, max_lag)
        else:
            this_res, this_z_res = get_event_vol(name, event, exogen_z_score, market_frame[name], max_lag)
        res = res.append(this_res, ignore_index=True)
        z_res = z_res.append(this_z_res, ignore_index=True)
    pos_res = res[res['Type'] == 1]
    neg_res = res[res['Type'] == -1]

    pos_z_score = z_res[z_res.Type == 1].mean(axis=0)
    pos_z_score.name = 'AvgZScore'
    neg_z_score = z_res[z_res.Type == -1].mean(axis=0)
    neg_z_score.name = 'AvgZScore'

    pos_summary_stat = pos_res.describe()
    neg_summary_stat = neg_res.describe()

    # Append t-test and Wilcoxon rank test p-values to the bottom of the result data frames
    pos_res_stat_tests = pos_res[day_range]
    neg_res_stat_tests = neg_res[day_range]

    # if mode is vol, we use the days leading up to the event day as the benchmark and conduct two-sample, one-tailed
    # t-test to ascertain whether the volatilities after the event days are significantly higher than before the event.
    if mode == 'vol':
        # T-test


        # Wilcoxon Rank-sum test
        #pos_wilcoxon = st.wilcoxon(pos_res_stat_tests.loc[:, list(range(0, max_lag))].values,
        #                           pos_res_stat_tests.loc[:, list(range(-max_lag, 0))].values,
        #                           alternative='greater') # one-tailed test here
        #neg_wilcoxon = st.wilcoxon(neg_res_stat_tests.loc[:, list(range(0, max_lag))].values,
        #                           neg_res_stat_tests.loc[:, list(range(-max_lag, 0))].values,
        #                           alternative='greater')  # one-tailed test here
        #

        # Convert two tailed p-value to one-tailed p-value
        pos_pval = pd.Series(1., name='t-test', index=pos_res_stat_tests.columns)
        neg_pval = pd.Series(1., name='t-test', index=neg_res_stat_tests.columns)

        for i in day_range:
            if i >= -int(max_lag / 2):
                pos_t_test = st.ttest_1samp(pos_res_stat_tests.loc[:, i].values,
                                     popmean=np.nanmean(pos_res_stat_tests.loc[:, list(range(-max_lag, -int(max_lag/2)))].values),
                                      nan_policy='omit')
                neg_t_test = st.ttest_1samp(neg_res_stat_tests.loc[:, i].values,
                                      popmean=np.nanmean(neg_res_stat_tests.loc[:, list(range(-max_lag, -int(max_lag/2)))].values),
                                      nan_policy='omit')

                pos_pval[i] = pos_t_test[1]/2 if pos_t_test[0] > 0 else 1 - pos_t_test[1] / 2
                neg_pval[i] = neg_t_test[1]/2 if neg_t_test[0] > 0 else 1 - neg_t_test[1] / 2

        pos_summary_stat = pos_summary_stat.append(pos_pval)
        # here we divide the p-value by 2 to obtain the one tailed p value

        #wilcoxon_pos = pd.Series({i: st.wilcoxon(pos_res_stat_tests[i] - pos_pop_mean)[1]
        #                          for i in day_range}, name='wilcoxon', index=pos_res_stat_tests.columns)

        neg_summary_stat = neg_summary_stat.append(neg_pval)

    else:
        # 2-tailed t-test against the hypothesis that the abnormal return should be 0
        pos_summary_stat = pos_summary_stat.append(pd.Series(
            st.ttest_1samp(pos_res_stat_tests, popmean=0, nan_policy='omit')[1], name='t-test', index=pos_res_stat_tests.columns))
        #wilcoxon_pos = pd.Series({i: st.wilcoxon(pos_res_stat_tests[i] - 0)[1]
        #                      for i in day_range}, name='wilcoxon', index=pos_res_stat_tests.columns)
        # pos_summary_stat = pos_summary_stat.append(wilcoxon_pos)
        # pos_summary_stat = pos_summary_stat.append(pos_z_score)
        neg_summary_stat = neg_summary_stat.append(pd.Series(
            st.ttest_1samp(neg_res_stat_tests, popmean=0, nan_policy='omit')[1], name='t-test', index=neg_res_stat_tests.columns))
        #wilcoxon_neg = pd.Series({i: st.wilcoxon(neg_res_stat_tests[i] - 0)[1] for i in day_range}, name='wilcoxon'
        #                      , index=neg_res_stat_tests.columns)
        #neg_summary_stat = neg_summary_stat.append(wilcoxon_neg)
        neg_summary_stat = neg_summary_stat.append(neg_z_score)

    if save_csv:
        file_appendix =  str(names[0]) + "_rollWindow" + str(rolling_window) + \
                            "_maxLag" + str(max_lag) + "_mode" + str(mode) + "_" + str(start_date.date()) + \
                            "_" + str(end_date.date()) + ".csv"
        pos_summary_stat.to_csv('output/posEvents' + file_appendix)
        neg_summary_stat.to_csv('output/negEvents' + file_appendix)

    return pos_summary_stat, neg_summary_stat, pos_res, neg_res


def network_event_analyser(leading_name, lagging_names, market_frame, exogen_frame, rolling_window=90,
                           detection_threshold=2.5, max_lag=5, mode='rtn', start_date=None, end_date=None):

    def process_sub_result(df, z_df):
        # print(z_df)
        groupby = df.groupby('Name')
        groupby_z = z_df.groupby('Name')
        res = {}
        for name, frame in groupby:
            data_stat_test = frame[day_rng]
            mean_z = groupby_z[day_rng].get_group(name).mean(axis=0)
            mean_z.name = 'AvgZScore'
            summary_stat = frame.describe()
            pop_mean = np.nanmean(data_stat_test.loc[:, list(range(-2*max_lag, -1))].values) if mode == 'vol' else 0
            ttest = pd.Series(st.ttest_1samp(data_stat_test, popmean=pop_mean, nan_policy='omit')[1], name='t-test', index=day_rng)
            wilcoxon = pd.Series({i: st.wilcoxon(data_stat_test[i] - pop_mean)[1] for i in day_rng}, name='wilcoxon', index=day_rng)
            summary_stat = summary_stat.append(ttest)
            summary_stat = summary_stat.append(wilcoxon)
            # print(mean_z)
            summary_stat = summary_stat.append(mean_z)
            res[name] = summary_stat

        # Average exclude leading name:
        df = df[df.Name != leading_name]
        data_stat_test = df[day_rng]
        mean_z = data_stat_test.mean(axis=0)
        mean_z.name = 'AvgZScore'
        summary_stat = df.describe()

        pop_mean = np.nanmean(data_stat_test.loc[:, list(range(-2 * max_lag, -1))].values) if mode == 'vol' else 0
        ttest = pd.Series(st.ttest_1samp(data_stat_test, popmean=pop_mean, nan_policy='omit')[1], name='t-test', index=day_rng)
        tstats = pd.Series(st.ttest_1samp(data_stat_test, popmean=pop_mean, nan_policy='omit')[0], name='t-stat', index=day_rng)
        wilcoxon = pd.Series({i: st.wilcoxon(data_stat_test[i]-pop_mean)[1] for i in day_rng}, name='wilcoxon', index=day_rng)
        wstats = pd.Series({i: st.wilcoxon(data_stat_test[i]-pop_mean)[0] for i in day_rng}, name='wstats', index=day_rng)
        summary_stat = summary_stat.append(ttest)
        summary_stat = summary_stat.append(wilcoxon)
        summary_stat = summary_stat.append(mean_z)
        summary_stat = summary_stat.append(tstats)
        summary_stat = summary_stat.append(wstats)

        res['Average'] = summary_stat
        return res

    if mode == 'rtn':
        day_rng = [i for i in range(-max_lag, max_lag+1)]
    else:
        day_rng = [i for i in range(-2*max_lag, max_lag)]

    start_date = pd.Timestamp(start_date) if start_date and pd.Timestamp(start_date) > market_frame.index[0] else market_frame.index[0]
    end_date = pd.Timestamp(end_date) if end_date and pd.Timestamp(end_date) < market_frame.index[-1] else market_frame.index[-1]

    market_frame = market_frame[(market_frame.index >= start_date) & (market_frame.index <= end_date)]
    exogen_frame = exogen_frame[(exogen_frame.index >= start_date) & (exogen_frame.index <= end_date)]

    exogen_series = exogen_frame[leading_name]
    exogen_z_score = _get_z_score(exogen_series, rolling_window)
    event = _get_event_series(exogen_z_score, detection_threshold)
    # Get the sentiment event for the LEADING NAME
    res = pd.DataFrame(columns=['Name', 'EventDate', 'Type'] + day_rng)
    z_res = pd.DataFrame(columns=res.columns)

    #print(market_frame)
    #print(exogen_frame)

    for lagging_name in lagging_names:
        daily_abnormal_return = pd.Series(np.nan, index=market_frame.index)
        z_score = _get_z_score(exogen_frame[lagging_name], rolling_window)
        if mode == 'rtn':
            for date in market_frame.index:
                if date < start_date + pd.to_timedelta(rolling_window, 'D'):
                    continue
                daily_abnormal_return[date] = get_abnormal_return(lagging_name, market_frame, date, window=180)
            this_res, this_z_res = get_event_cumulative_abnormal_return(lagging_name, event, z_score, daily_abnormal_return, max_lag)
        else:
            this_res, this_z_res = get_event_vol(lagging_name, event, z_score, market_frame[lagging_name], max_lag)
        res = res.append(this_res, ignore_index=True)
        # print(this_z_res)
        z_res = z_res.append(this_z_res, ignore_index=True)

    pos = res[res['Type'] == 1]
    neg = res[res['Type'] == -1]
    pos_z = z_res[z_res.Type == 1]
    neg_z = z_res[z_res.Type == -1]
    print(pos)
    print(neg)

    pos_res = process_sub_result(pos, pos_z)
    neg_res = process_sub_result(neg, neg_z)
    print(pos_res)
    print(neg_res)

    return pos_res, neg_res


