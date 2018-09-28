import utilities
from source.utils import cosine_distance
import networkx as nx
import community
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from visualiser import get_color_map


class NetworkAnalyser:
    def __init__(self, full_data_object, names, start_date=None, end_date=None):
        self.names = names
        self.start_date = start_date if start_date and start_date >= full_data_object.start_date else full_data_object.start_date
        self.end_date = end_date if end_date and end_date <= full_data_object.end_date else full_data_object.end_date
        if self.start_date != full_data_object.start_date or self.end_date != full_data_object.end_date:
            self.full_data = utilities.create_sub_obj(full_data_object, self.start_date, self.end_date)
        else:
            self.full_data = full_data_object
        self.G = self.full_data.build_occurrence_network_graph(focus_iterable=self.names)
        self.partition = {}
        self.node_pos = None

    def get_centrality(self, names=None, type='katz', ordered=True):
        if type == 'katz':
            centrality = nx.katz_centrality(self.G, weight='weight')
        elif type == 'eigen':
            centrality = nx.eigenvector_centrality(self.G, weight='weight')
        else:
            raise NotImplementedError('Unrecognised centrality mode name.')
        if ordered:
            centrality = sorted(centrality.items(), key=lambda kv: -kv[1])
        if names is not None:
            res = [x for x in centrality if x[0] in names]
            return res
        else: return centrality

    def cluster(self, resolution=None):
        if resolution:
            self.partition = community.best_partition(self.G, weight='weight', resolution=resolution)
        else:
            self.partition = community.best_partition(self.G, weight='weight')
        return self.partition

    def get_cluster_peers(self, name):
        if self.partition == {}:
            self.cluster()
        peers = [key for key, value in self.partition.items() if value == self.partition[name]]
        peer_centrality = self.get_centrality(names=peers)
        res = {peer[0]: peer[1] for peer in peer_centrality}
        return res

    def get_category(self, name):
        if self.partition == {}:
            self.cluster()
        return self.partition[name]

    def get_neighbours(self, name, max_neighbour=-1, weight_threshold=-1, exclude=None):
        neighbours = self.G[name]
        sorted_neighbours = sorted(neighbours.items(), key=lambda kv: -kv[1]['weight'])
        #print(sorted_neighbours)
        if weight_threshold > 0:
            if isinstance(weight_threshold, float):
                res = [item for item in sorted_neighbours if item[1]['weight'] >= weight_threshold]
            elif isinstance(weight_threshold, int):
                threshold = np.percentile(np.array([x[1]['weight'] for x in sorted_neighbours]), weight_threshold)
                res = [item for item in sorted_neighbours if item[1]['weight'] >= threshold]
            else:
                raise TypeError("percentile threshold needs to be a positive integer or float")
        else:
            res = sorted_neighbours
        if neighbours != -1:
            res = res[:min(max_neighbour, len(sorted_neighbours))]
        if exclude is not None:
            res = [item for item in res if item[0] not in exclude]
        names = [name] + [each_res[0] for each_res in res]
        weight = [1] + [each_res[1]['weight'] for each_res in res]
        category = [self.get_category(name)] + [self.get_category(each_name) for each_name in names]
        centrality = [self.get_centrality(name)[0][1]] + [self.get_centrality(each_name)[0][1] for each_name in names]
        res = pd.DataFrame([names, weight, category, centrality]).transpose()
        res.columns = ['Names', 'CosineDistance', 'Category', 'Centrality']
        res.dropna(axis=0, inplace=True, how='any')
        return res

    def plot_network(self, group_by='category', threshold=90):
        if self.partition == {}: self.cluster()
        color_map = get_color_map(self.names, self.partition)
        c = [color_map[name] for name in self.G.nodes()]
        if self.node_pos is None:
            pos = nx.spring_layout(self.G)
            self.node_pos = pos
        edgelist = self.G.edges(data=True)
        if group_by == 'centrality':
            centrality = list(self.get_centrality().values())
            vmin = np.min(centrality)
            vmax = np.max(centrality)

        if threshold:
            if isinstance(threshold, int):
                weight_list = np.array([x[2]['weight'] for x in edgelist])
                threshold = np.percentile(weight_list, threshold)
                edgelist = [x for x in edgelist if x[2]['weight'] >= threshold]
            else:
                edgelist = [x for x in edgelist if x[2]['weight'] >= threshold]
        if group_by == 'category':
            nx.draw(self.G, self.node_pos, edgelist=edgelist, node_color=c, with_labels=True)
        elif group_by == 'centrality':
            nx.draw(self.G, self.node_pos, edgelist=edgelist,
                    cmap=plt.plasma(),
                    node_color=centrality,
                    with_labels=True)
            sm = plt.cm.ScalarMappable(cmap=plt.plasma(), norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            plt.colorbar(sm)
        else:
            raise NotImplemented()

    def plot_single_name(self, name, max_neighbour=-1, weight_threshold=-1):
        if self.partition == {}: self.cluster()
        data = self.get_neighbours(name, max_neighbour, weight_threshold)
        nodes = data.iloc[:, 0]
        G = nx.Graph()
        G.add_nodes_from(nodes)
        nx.draw(G)

    def get_cos_dist_time_series(self, name1, *names, correlation_period=90):
        entity_list = [name1] + list(names)
        name2id = {entity_list[i]: i for i in range(len(entity_list))}
        occurrence_array = np.zeros((len(entity_list), self.full_data.news_number))
        date_array = np.zeros((self.full_data.news_number, ))
        start_date = self.start_date + pd.to_timedelta(correlation_period)
        res = {entity: [] for entity in names}
        date_frame = []
        news_id = 0
        day_id = 0
        for each_day in self.full_data.days:
            for news in each_day.day_news:
                for entity in entity_list:
                    if entity in news.entity_occur:
                        occurrence_array[name2id[entity], news_id] = news.entity_occur[entity]
                        date_array[news_id] = day_id
                news_id += 1
            if each_day.date >= start_date + pd.to_timedelta(correlation_period):
                date_frame.append(each_day.date)
                included_news = (date_array >= day_id - correlation_period) & (date_array < day_id)
                sub_occurrence_array = occurrence_array[:, included_news]
                for name in names:
                    res[name].append(cosine_distance(sub_occurrence_array[name2id[name1]],
                                                     sub_occurrence_array[name2id[name]]))
            day_id += 1
        date_array = pd.Series(date_frame)
        res = pd.DataFrame(res, index=date_array)
        return res

    def get_dynamic_neighbors(self, name, date, period=180, weight_threshold=-1, max_neighbour=-1, exclude=None):
        """
        Get top n neighbours in based on cosine distance in the past n days
        :param name:
        :param time:
        :param period:
        :param weight_threshold:
        :param max_neighbour:
        :param exclude:
        :return:
        """
        if date < self.start_date + pd.to_timedelta(period):
            return np.nan
        #print(date - pd.to_timedelta(period, 'D', ), date)
        day_objs = [day for day in self.full_data.days if ((day.date > date - pd.to_timedelta(period, 'D')) and (day.date <= date))]
        name2id = {self.names[i]: i for i in range(len(self.names))}
        news_cnt = sum([day.news_number for day in day_objs])
        res = {entity: 0 for entity in self.names}
        occurrence_array = np.zeros((len(self.names), news_cnt))
        news_id = 0
        for each_day in day_objs:
            for news in each_day.day_news:
                for entity in self.names:
                    if entity in news.entity_occur:
                        occurrence_array[name2id[entity], news_id] = news.entity_occur[entity]
                news_id += 1

        for each_name in self.names:
            res[each_name] = cosine_distance(occurrence_array[name2id[name]], occurrence_array[name2id[each_name]])
        res = sorted(res.items(), key=lambda v: -v[1])[1:]

        if weight_threshold != -1:
            res = [x for x in res if x[1] > weight_threshold]
        if max_neighbour != -1 and max_neighbour < len(res):
            thres = res[max_neighbour][1]
            res = [x for x in res if x[1] > thres]
        if exclude is not None:
            res = [x for x in res if x[0] not in exclude]
        print(res)
        return res

