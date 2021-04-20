import torch

from Logic.ProperLogic.cluster_modules.cluster import Cluster


class EvalCluster(Cluster):
    metric = 2

    @classmethod
    def set_metric(cls, metric):
        cls.metric = metric

    @classmethod
    def compute_dist(cls, embedding1, embedding2):
        return float(torch.dist(embedding1, embedding2, p=cls.metric))
