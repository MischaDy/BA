import operator
from itertools import chain

from Logic.ProperLogic.helper_classes.reducer import MaxReducer
from Logic.ProperLogic.misc_helpers import log_error


class ClusterDict(dict):
    # TODO: Make sure constructor is only called when needed / doesn't produce more work than necessary!

    def __init__(self, clusters=None):
        super().__init__()
        self.max_id = None
        self.max_id_reducer = MaxReducer()

        if not clusters:
            return

        for cluster in clusters:
            cluster_id = cluster.cluster_id
            self[cluster_id] = cluster
            self.max_id_reducer(cluster_id)
        self.max_id = self.max_id_reducer.get_state()

    def get_clusters(self, with_ids=False):
        if with_ids:
            return self.items()
        return self.values()

    def get_cluster_by_id(self, cluster_id):
        try:
            return self[cluster_id]
        except KeyError:
            log_error(f"no cluster with id '{cluster_id}' found")
            return None

    def get_clusters_by_ids(self, cluster_ids):
        return map(self.get_cluster_by_id, cluster_ids)

    def get_cluster_ids(self):
        return self.keys()

    def get_cluster_labels(self, with_ids=False, unique=True):
        """
        If with_ids is provided, unique is ignored.

        :param with_ids:
        :param unique:
        :return:
        """
        attrs = ['cluster_id'] if with_ids else []
        attrs.append('label')
        cluster_labels = self.get_cluster_attrs(*attrs)

        if unique and not with_ids:
            return list(set(cluster_labels))
        return list(cluster_labels)

    def get_cluster_attrs(self, *attrs):
        clusters = self.get_clusters()
        attrs_getter = operator.attrgetter(*attrs)
        return map(attrs_getter, clusters)

    def reset_ids(self, start_id=1):
        clusters_with_ids = list(self.get_clusters(with_ids=True))
        self.clear()
        old_ids = []
        for new_cluster_id, (old_cluster_id, cluster) in enumerate(clusters_with_ids, start=start_id):
            old_ids.append(old_cluster_id)
            cluster.set_cluster_id(new_cluster_id)
            self[new_cluster_id] = cluster

        max_id = start_id + len(clusters_with_ids) - 1
        new_ids = list(range(start_id, max_id))
        self.max_id = max_id
        return old_ids, new_ids

    def set_ids(self, old_ids, new_ids):
        clusters = self.get_clusters()
        old_to_new_ids_dict = dict(zip(old_ids, new_ids))
        self.max_id_reducer.reset()
        for cluster in clusters:
            new_id = old_to_new_ids_dict[cluster.cluster_id]
            cluster.set_cluster_id(new_id)
            self.max_id_reducer(new_id)
        self.max_id = self.max_id_reducer.get_state()

    def any_cluster_with_emb(self, emb):
        clusters = self.get_clusters()
        return any(filter(lambda cluster: cluster.contains_embedding(emb), clusters))

    def add_clusters(self, clusters):
        self.max_id_reducer.reset()
        for cluster in clusters:
            cluster_id = cluster.cluster_id
            self[cluster_id] = cluster
            self.max_id_reducer(cluster_id)
        self.max_id = self.max_id_reducer.get_state()

    def add_cluster(self, cluster):
        self.add_clusters([cluster])

    def remove_clusters(self, clusters):
        reset_max_id = False
        for cluster in clusters:
            cluster_id = cluster.cluster_id
            self.pop(cluster_id)
            if cluster_id == self.max_id:
                reset_max_id = True
        if reset_max_id:
            self.reset_max_id()

    def reset_max_id(self):
        cluster_ids = self.get_cluster_ids()
        self.max_id = max(cluster_ids) if cluster_ids else 0

    def remove_cluster(self, cluster):
        self.remove_clusters([cluster])

    def get_max_id(self):
        if self.max_id is None:
            return self.max_id_reducer.default
        return self.max_id

    def get_embeddings(self):
        return chain(*map(lambda cluster: cluster.get_embeddings(),
                          self.get_clusters(with_ids=False)))
