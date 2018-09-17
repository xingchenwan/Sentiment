# Xingchen Wan | Oxford-Man Institute of Quantitative Finance | Updated 6 Sep 2018

import processor, grouper, analyser, visualiser, utilities
from source.all_classes import *
import matplotlib.pylab as plt
import pandas as pd

if __name__ == "__main__":

    full_data_obj = 'full.date.20061020-20131120'
    market_data_path = 'data/top_entities.xlsx'
    market_data_sheet = 'Sheet3'
    sector_path = "source/sector.csv"

    with open(full_data_obj, 'rb') as f:
        all_data = pickle.load(f, fix_imports=True, encoding='bytes')
    utilities.fix_fulldata(all_data)

    start_date = datetime.date(2006, 10, 1)
    end_date = datetime.date(2013, 12, 31)

    sub_all_data = utilities.create_sub_obj(all_data, start_date, end_date)
    price, daily_rtn, log_return, vol = processor.process_market_time_series(market_data_path, market_data_sheet,
                                                      start_date=pd.to_datetime(start_date),
                                                      end_date=pd.to_datetime(end_date),
                                                      )

    #dynamic = grouper.get_dynamic_grouping(all, sub_all_data)
    #visualiser.plot_network(sub_all_data, names=all, categories=dynamic)
    #plt.show()


    this_name = 'GPS'
    all_names = list(price.columns)
    sector = grouper.get_sector_grouping(all_names, 'source/sector.csv', 0, 2)
    this_sector = grouper.get_sector_peer(sector, this_name)

    count, med_sentiment, sum_sentiment = processor.process_count_sentiment(all_data,
                                                                            start_date=pd.to_datetime(start_date),
                                                                            end_date=pd.to_datetime(end_date),
                                                                            focus_iterable=all_names,
                                                                            rolling=True,
                                                                            rolling_smoothing_factor=0.7)

    pos, neg = analyser.event_analyser(this_sector, daily_rtn, sum_sentiment, max_lag=5, rolling_window=90, detection_threshold=2.5)
    pos.iloc[1, :].plot()
    neg.iloc[1, :].plot()
    plt.axvline(x=-1)
    plt.axhline(y=0)
    plt.show()
    exit()
    ################################################

    res = analyser.rolling_correlate(this_name, 180, med_sentiment, sum_sentiment, log_return, count, vol,
                                     max_lag=2,
                                     frame_names=['med_sentiment', 'sum_sentiment', 'log_return', 'count', 'vol', ]
                                     , include_pairs=[('log_return', 'count'), ('log_return', 'med_sentiment'),
                                                      ('log_return', 'sum_sentiment'),
                                                      ('vol', 'count')]
                                     , count_threshold={'count': 0})
    visualiser.plot_single_name(this_name, *[res[i] for i in list(res.columns)], log_return, med_sentiment,
                                sum_sentiment, price, vol, count,
                                arg_names=list(res.columns) + ['logreturn', 'med_sentiment', 'sum_sentiment', 'px', 'vol','count'])
    plt.show()
    exit()

    ##################################################

    price[this_name].plot()
    sum_sentiment[this_name].plot()
    plt.show()

    res = analyser.correlate(log_return, vol, med_sentiment, sum_sentiment,
                             count, max_lag=2, plot=False, save_csv=True, display_summary_stats=True,
                             focus_iterable=all, count_threshold=0)

    # stat_tests(res['PriceCountCorr'])
    analyser.stat_tests(res['PriceMedSentimentCorr'])
    analyser.stat_tests(res['PriceSumSentimentCorr'])
    analyser.stat_tests(res['VolCountCorr'])
    # stat_tests(res['SpotVolCorr'])

    # dynamic_group = grouper.get_dynamic_grouping(res['Name'], all_data, start_date, end_date)
    #visualiser.plot_scatter(res['PriceCountCorr'], res['Name'],
    #                        x_label='Stock Names', y_label='Correlation Coefficient', categories=sector)

    plt.show()
    exit()
    ##################################################
    #analyser.anomaly_correlate(this_name, 365, sum_sentiment, log_return, anomaly_threshold=2.5)
    #plt.show()




