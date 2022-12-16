# -*- coding: utf-8 -*-
from sklearn.cluster import KMeans
import numpy as np
from numpy.typing import ArrayLike
"""Main module."""


class StratifiedClusterer:

    strata_kmeans = None
    strata = None

    @property
    def n_strata(self):

        assert self.strata is not None, "Strata are not yet defined!"

        return len(self.strata) + 2

    @staticmethod
    def validate_data(data, stratify_data):

        assert len(data) == len(stratify_data), "Number of datapoint and stratification points do not agree!"

    def assign_strata(self, stratify_data):

        assert self.strata is not None, "Attempting to assign to strata, but strata have not been defined yet!"

        stratum_assignments = np.digitize(stratify_data, self.strata)

        return stratum_assignments

    def fit(self, data: ArrayLike, k: int, stratify_coordinate: ArrayLike, strata: ArrayLike):

        self.validate_data(data, stratify_coordinate)

        # Assign each stratify_coordinate to a stratum
        # TODO: For now, assuming the strata are ordered
        assert strata == np.sort(strata), "Strata boundaries not sorted!"
        self.strata = strata

        stratum_assignments = self.assign_strata(stratify_coordinate)

        # For each stratum, go through, pick out all the coordinates in it, and cluster on them
        # The total number of strata is the number of provided strata + 2 (the +2 add bins going to -inf and +inf)
        # TODO: Handle if the user already has bounds going from -inf to +inf
        self.strata_kmeans = [
            KMeans(n_clusters=k)
            for _ in range(self.n_strata)
        ]

        for stratum_index in range(self.n_strata):

            stratum_kmeans = self.strata_kmeans[stratum_index]

            points_in_stratum = np.argwhere(stratum_assignments == stratum_index)
            data_in_stratum = data[points_in_stratum]

            stratum_kmeans.fit(data_in_stratum)

    def predict(self, data, stratify_coordinate):

        self.validate_data(data, stratify_coordinate)

        assert self.strata_kmeans is not None, "Attempting to predict, but stratum KMeans models have not been fit()!"

        stratum_assignments = self.assign_strata(stratify_coordinate)

        # Each stratum will assign cluster IDs that are indexed from 0, within that stratum.
        # In order to get globally unique cluster numbers, we'll just keep a counter. So, if we have 5 clusters in
        #   stratum 0, then the 0th cluster in stratum 1 will be assigned id 5.
        stratum_cluster_offset = 0

        cluster_assignments = np.zeros(data.shape[:2], dtype=int)

        for stratum_index in range(self.n_strata):

            stratum_kmeans = self.strata_kmeans[stratum_index]

            points_in_stratum = np.argwhere(stratum_assignments == stratum_index)
            data_in_stratum = data[points_in_stratum]

            points = stratum_kmeans.predict(data_in_stratum) + stratum_cluster_offset
            cluster_assignments[points_in_stratum] = points

            stratum_cluster_offset += len(stratum_kmeans.cluster_centers_)

        assert len(cluster_assignments) == len(data)

        return cluster_assignments
