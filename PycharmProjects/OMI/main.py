# Xingchen Wan | Oxford-Man Institute of Quantitative Finance | Updated 6 Sep 2018

import processor, grouper, analyser, visualiser, utilities
from source.all_classes import *
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pylab as plt
import pandas as pd


NAME = 'USDJPY'
MODE = 'vol'
# Sector representatives:
# Financials: AXP
# Tech: AAPL
# Retail: WMT
# Industrials: BA
START_DATE = datetime.date(2007, 1, 1)
END_DATE = datetime.date(2007, 12, 31)
MAX_LAG = 10

PATH_POS = "output/" + "posEvents" + NAME + "_rollWindow180_maxLag" + str(MAX_LAG) + "_mode" + MODE + "_" + \
           str(START_DATE) + "_" + str(END_DATE) + ".csv"
PATH_NEG = "output/" + "negEvents" + NAME + "_rollWindow180_maxLag" + str(MAX_LAG) + "_mode" + MODE + "_" + \
           str(START_DATE) + "_" + str(END_DATE) + ".csv"
FIGURE_PATH = "figures/" + NAME + "_rollWindow180_maxLag" + str(MAX_LAG) + "_mode" + MODE + "_" + \
           str(START_DATE) + "_" + str(END_DATE) + ".png"


def plot_from_csv():
    pos_df = pd.read_csv(PATH_POS, index_col=0)
    neg_df = pd.read_csv(PATH_NEG, index_col=0)
    visualiser.plot_events(pos_df, neg_df, descps=['Positive Events', 'Negative Events'], mode=MODE)
    plt.xlabel("Number of days to events")
    # plt.savefig(FIGURE_PATH)


def plot_multiple_from_csv(paths: dict, plot_range=None):
    """
    Plot multiple graphs from a collection of csv paths. the paths should be organised in a dictionary ordered in a
    chronological order. The key is the year and the value is the path
    :param paths:
    :return:
    """

    from collections import OrderedDict

    # setting colours

    def plot_event(event_summary, year, mode='vol', alpha=0.1, plot_range=None):
        ubound = 0.1
        lbound = 0


        event_summary = event_summary[[i for i in list(event_summary.columns) if i.lstrip('-').isdigit()]]
        event_summary.columns = event_summary.columns.astype(int)

        if plot_range is None:
            xi = np.array((event_summary.columns))

        else:
            xi = np.array([x for x in range(plot_range[0], plot_range[1])])

        mean_val = event_summary.loc['mean', xi]
        offset = mean_val.iloc[0] if mode != 'vol' else 0
        p = plt.plot(xi + 1 , event_summary.loc['mean', xi] - offset, 'o-', linewidth=1, alpha=1, label=str(year), markersize=2)
        for i in range(len(xi)):
            if event_summary.loc['mean', xi[i]] - offset > ubound:
                plt.plot(xi[i] + 1, ubound * 0.97, marker="^", color=p[0].get_color())
            elif event_summary.loc['mean', xi[i]] - offset < lbound:
                plt.plot(xi[i] + 1, lbound * 0.97, marker='v', color=p[0].get_color())
        #test_stats = event_summary.loc['t-test', :]

        #sig_idx = np.argwhere((test_stats < 0.05) & (test_stats > 0.01)).reshape(-1) + mean_val.index[0]
        #v_sig_idx = np.argwhere(test_stats <= 0.01).reshape(-1).reshape(-1)+ mean_val.index[0]

        #if len(sig_idx):
        #    plt.plot(sig_idx, mean_val[sig_idx] - offset, 'o', color='orange', label='p < 0.05')
        #if len(v_sig_idx):
        #    plt.plot(v_sig_idx, mean_val[v_sig_idx] - offset, 'o', color='red', label='p < 0.01')
        if mode == 'rtn':
            plt.ylabel('Cumulative Abnormal Return (CAR)')
        elif mode == 'vol':
            plt.ylabel('Volatility 1D')

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.grid(True)
        plt.gca().set_ylim(bottom=lbound)
        plt.gca().set_ylim(top=ubound)
        plt.legend(by_label.values(), by_label.keys(), prop={'size': 6}, loc='upper right')

    i = 0
    plt.axvspan(-1, 0, alpha=0.2, color='gray')
    plt.rcParams.update({'font.size': 8})

    for year, path_pos in paths.items():
        i += 1
        pos_df = pd.read_csv(path_pos, index_col=0)
        plot_event(pos_df, year, mode=MODE, plot_range=plot_range)
        plt.xlabel("Number of days to events")
    plt.show()



def event(start_date=START_DATE, end_date=END_DATE):

    full_data_obj = 'data/full.date.20061020-20131120'
    market_data_path = 'data/market_data.xlsx'
    market_data_sheet = 'Price1'
    volume_data_sheet = 'Volatility1'
    sector_path = "source/sector.csv"
    pd.options.display.max_rows = 10000

    with open(full_data_obj, 'rb') as f:
        all_data = pickle.load(f, fix_imports=True, encoding='bytes')
    utilities.fix_fulldata(all_data)

    sub_all_data = utilities.create_sub_obj(all_data, start_date, end_date)
    price, daily_rtn, log_return, vol = processor.process_market_time_series(market_data_path, market_data_sheet,
                                                      start_date=pd.to_datetime(start_date),
                                                      end_date=pd.to_datetime(end_date),
                                                      )
    volume, _, _, _ = processor.process_market_time_series(market_data_path,volume_data_sheet,
                                                           start_date=pd.to_datetime(start_date),
                                                           end_date= pd.to_datetime(end_date))

    all_names = list(price.columns)[:-1]
    sector = grouper.get_sector_grouping(all_names, sector_path, 0, 2)
    this_sector = grouper.get_sector_peer(sector, NAME)

    count, med_sentiment, sum_sentiment = processor.process_count_sentiment(sub_all_data,
                                                                            start_date=pd.to_datetime(start_date),
                                                                            end_date=pd.to_datetime(end_date),
                                                                            focus_iterable=all_names,
                                                                            rolling=True,
                                                                            rolling_smoothing_factor=0.7)

    pos, neg, _, _ = analyser.event_analyser(all_names, market_frame=vol, exogen_frame=sum_sentiment, mode=MODE,
                                             start_date=start_date,
                                             end_date=end_date,
                                             max_lag=MAX_LAG)


def event_by_year(start_date, end_date):
    for _ in range(7):
        print(start_date, end_date)
        event(start_date, end_date)
        start_date += datetime.timedelta(days=365)
        end_date += datetime.timedelta(days=365)



if __name__ == "__main__":
    #event_by_year(START_DATE, END_DATE)

    event_type = "neg"

    paths_dict = {
        '2007': "output/" + event_type + "Events" + NAME + "_rollWindow180_maxLag10_mode" + MODE + "_2007-01-01_2007-12-31" + ".csv",
        '2008': "output/" + event_type + "Events" + NAME + "_rollWindow180_maxLag10_mode" + MODE + "_2008-01-01_2008-12-30" + ".csv",
        '2009': "output/" + event_type + "Events" + NAME + "_rollWindow180_maxLag10_mode" + MODE + "_2008-12-31_2009-12-30" + ".csv",
        '2010': "output/" + event_type + "Events" + NAME + "_rollWindow180_maxLag10_mode" + MODE + "_2009-12-31_2010-12-30" + ".csv",
        '2011': "output/" + event_type + "Events" + NAME + "_rollWindow180_maxLag10_mode" + MODE + "_2010-12-31_2011-12-30" + ".csv",
        '2012': "output/" + event_type + "Events" + NAME + "_rollWindow180_maxLag10_mode" + MODE + "_2012-01-02_2012-12-28" + ".csv",
        '2013': "output/" + event_type + "Events" + NAME + "_rollWindow180_maxLag10_mode" + MODE + "_2012-12-31_2013-12-27" + ".csv"
    }

    paths_dict2 = {
        'Positive': "output/" + "pos" + "Events" + NAME + "_rollWindow180_maxLag10_modevol_2006-10-30_2013-12-31" + ".csv",
        'Negative': "output/" + "neg" + "Events" + NAME + "_rollWindow180_maxLag10_modevol_2006-10-30_2013-12-31" + ".csv",
    }
    plot_multiple_from_csv(paths_dict, plot_range=[-10, 5])




