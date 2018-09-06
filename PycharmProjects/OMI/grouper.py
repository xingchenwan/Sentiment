# Grouper module: generate different classification of the entities, either based on dynamic cosine distance in relevant
# co-occurrence matrices or by static sector categorisation

import datetime
import pandas as pd
from source.all_classes import *
import community
from utilities import *


def get_dynamic_grouping(names, full_data_obj, start_date=None, end_date=None):
    """
    Compute the dynamic grouping dictionary of a series of single names based on the cosine-distance of the entities
    within the specified period of time
    :param start_date: optional. If none, the start date will be taken to be the same as the starting date of the
    full_data object
    :param end_date: optional. Same as above
    :return: Dict mapping names to their dynamic grouping
    """
    if start_date and not isinstance(start_date, datetime.date):
        raise TypeError("Start time needs to be a date object")
    if end_date and not isinstance(end_date, datetime.date):
        raise TypeError("End time needs to be a date object")
    if not isinstance(full_data_obj, FullData):
        raise TypeError("full_data_obj needs to be a FullData object")
    if isinstance(names, pd.Series):
        names = names.values.tolist()
    start_date = start_date if start_date and start_date >= full_data_obj.start_date else full_data_obj.start_date
    end_date = end_date if end_date and end_date <= full_data_obj.end_date else full_data_obj.end_date
    # Sanity Checks

    full_data_sub = create_sub_obj(full_data_obj, start_date, end_date)
    nx_graph = full_data_sub.build_occurrence_network_graph(focus_iterable=names)
    partition = community.best_partition(nx_graph, resolution=7)
    return partition


def get_sector_grouping(names, path, name_col, sector_col):
    """
    Get the ground truth static sector grouping dictionary of a series of single names. If there are any names in
    'names' argument that do not have a sector grouping in the lookup csv file, these names will be assigned an
    "unclassified" grouping in the output dictionary
    :param path: str. path string of the csv file containing the sector information
    :param name_col: str or int. If str, this argument is interpreted as the name of the column. If int,
    this arguement is interpreted as the column number
    :param sector_col: Same as above for the sector column
    :return: Dict mapping names to their sector grouping
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

