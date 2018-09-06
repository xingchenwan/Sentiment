# Analyser: modules that conduct numerical analysis on the result
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pylab as plt


def correlate(market_data, vol_data, sentiment_med_data, sentiment_sum_data, count_data, max_lag=2,
              plot=True, save_csv=False,
              display_summary_stats=True):

    def normalised_corr(series1, series2, max_period=max_lag):
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
    benchmark_data = market_data['_ALL'] # MSCI log-return

    # market_data = market_data.subtract(benchmark_data, axis='index')
    # Superior log return over SPX return
    vol = vol_data.loc[intersect_dates].astype('float64')
    # market_data_bin = market_data.apply(lambda x: np.sign(x-x.shift(1)))[1:]

    # Compute the log-return of the stock prices over the previous period. The first data is removed because it is a NaN
    res = []

    for single_name in list(market_data.columns):
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
        all_nan = True if np.isnan(market_series).all() else False
        # Flag that the entire numpy series consists of NaN

        if not all_nan:
            market_series = market_series[non_nan_idx]
            count_series = count_series[non_nan_idx]
            sentiment_series_med = sentiment_series_med[non_nan_idx]
            sentiment_series_sum = sentiment_series_sum[non_nan_idx]
            vol_series = vol_series[non_nan_idx]
            # Then prune the data to get rid of the NaNs

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


def get_top_n_entities(FullData_obj, n=200):
    """
    Return a dictionary of the entities that appear in the top-n ranks in the news covered in the FullData object.
    Key is the (abbreviated) names of the entities and Value is the number of times the entities appeared
    :param FullData_obj: a FullData Object
    :param n:
    :return:
    """
    return dict(sorted(FullData_obj.entity_occur_interval.items(), key=lambda x: x[1])[-n:])

