# Utilities module: provide some utilities functions
import datetime
from source.all_classes import *


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


def create_sub_obj(parent_obj, start_date, end_date):
    """
    Create a new sub FullData object from an existing FullData object with a narrower range of dates.
    :param parent_obj: FullData object. The parent object
    :param start_date: datetime.date type. Start date of the sub object. Must be later than the parent start date
    :param end_date: datetime.date type. End date fo the sub object. Must be earlier than the parent end date.
    :return:
    """
    if not isinstance(parent_obj, FullData):
        raise TypeError("The parent_obj must be f FullData Type")
    if not isinstance(start_date, datetime.date) or start_date < parent_obj.start_date:
        raise ValueError("Start date needs to be of date type and should be later than the parent object start date!")
    if not isinstance(end_date, datetime.date) or end_date > parent_obj.end_date:
        raise ValueError("End date needs to be of date type and should be earlier than parent object end date")
    copied_obj = copy.deepcopy(parent_obj)
    # Create a copied object of the parent object first

    copied_obj.entity_occur_interval = {}
    copied_obj.entity_sentiment_interval = {}
    copied_obj.start_date = start_date
    copied_obj.end_date = end_date
    # Reset various properties of the copied object

    copied_obj.days = [day for day in parent_obj.days if day.date > start_date and day.date < end_date]
    total_sentiment = {}
    for each_day in copied_obj.days:
        for entity, num in each_day.entity_occur_day.items():
            if entity in copied_obj.entity_occur_interval:
                copied_obj.entity_occur_interval[entity] += num
                total_sentiment[entity] += num * each_day.entity_sentiment_day[entity]
            else:
                copied_obj.entity_occur_interval[entity] = num
                total_sentiment[entity] = num * each_day.entity_sentiment_day[entity]
    for entity, num in copied_obj.entity_occur_interval.items():
        copied_obj.entity_sentiment_interval[entity] = total_sentiment[entity] / num
    return copied_obj
