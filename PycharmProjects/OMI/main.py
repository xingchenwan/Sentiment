import networkx as nx

from source.utils import *
from source.all_classes import *
import matplotlib.pylab as plt
from collections import Counter
import pandas as pd
import operator

def main():
    G = nx.Graph()


def fix_object(obj):
    """
    Decode the byte data type back to strings due to Python 3.x un-pickling
    :param obj: all_data object
    :return: None
    """
    obj.__dict__ = dict((k.decode("ascii"), v) for k, v in obj.__dict__.items())


def fix_fulldata(full_data):
    fix_object(full_data)
    for day in full_data.days:
        fix_object(day)
        for news in day.day_news:
            fix_object(news)

if __name__ == "__main__":
    save_full_name = 'full.date.20061020-20131120'
    with open(save_full_name, 'rb') as f:
        all_data = pickle.load(f, fix_imports=True, encoding='bytes')
    fix_fulldata(all_data)

    top_n = 400 # Retrieve top n number of the entities that have occurred the most
    top_n_entities = dict(sorted(all_data.entity_occur_interval.items(), key=lambda x: x[1])[-top_n:])
    # Dictionary of the top n mentioned entities

    start_date = all_data.start_date
    i = 0
    stop_date = all_data.end_date

    all_days = all_data.days
    period = 5  # Assuming every week has 5 workdays

    # Initialise the aggregate list
    aggregate_entity_count = []
    aggregate_sentiment_count = []

    # Initialise the weekly counters
    period_entity_count = Counter({})
    period_sentiment_count = Counter({})

    for i in range((stop_date - start_date).days - 1):
        current_day = all_data[i] # Retrieve the Day object of the current day
        if current_day.day_news == 0:
            continue
        else:
            if i % period == 0:
                aggregate_entity_count.append(period_entity_count)
                aggregate_sentiment_count.append(aggregate_sentiment_count)
                period_entity_count = entity_count
                period_sentiment_count = sentiment_count
            else:
                period_entity_count += entity_count
                period_sentiment_count += sentiment_count

    # print(aggregate_entity_count)






