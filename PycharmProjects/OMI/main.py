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

    start_date = datetime.date(2007, 6, 30)
    end_date = datetime.date(2007, 12, 31)
    price, _, log_return, vol = processor.process_market_time_series(market_data_path, market_data_sheet,
                                                      start_date=pd.to_datetime(start_date),
                                                      end_date=pd.to_datetime(end_date))

    count, med_sentiment, sum_sentiment = processor.process_count_sentiment(all_data,
                                                                            start_date=pd.to_datetime(start_date),
                                                                            end_date=pd.to_datetime(end_date),
                                                                            focus_iterable=list(price.columns))

    res = analyser.correlate(log_return, vol, med_sentiment, sum_sentiment,
                             count, max_lag=0, plot=False, save_csv=False, display_summary_stats=True)

    #stat_tests(res['PriceCountCorr'])
    #stat_tests(res['PriceMedSentimentCorr'])
    analyser.stat_tests(res['PriceSumSentimentCorr'])
    analyser.stat_tests(res['VolCountCorr'])
    #stat_tests(res['SpotVolCorr'])
    # sector = (grouper.get_sector_grouping(res['Name'],sector_path, 0, 2, ))

    dynamic_group = grouper.get_dynamic_grouping(res['Name'], all_data, start_date, end_date)

    visualiser.plot_scatter(res['VolCountCorr'], res['Name'],
                 x_label='Stock Names', y_label='Correlation Coefficient', categories=dynamic_group)

    plt.show()
