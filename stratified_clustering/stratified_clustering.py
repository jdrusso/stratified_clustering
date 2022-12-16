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
    def validate_data(data: ArrayLike, stratify_data: ArrayLike) -> None:
        """
        These should not necessarily have to be the same dimensionality, but they should be the same length.
        In other words, each frame can have 3-D stratified coordinates and 5-D data-coordinates, or vice versa, but
            the number of frames must match.
        """

        assert len(data) == len(stratify_data), "Number of datapoint and stratification points do not agree!"

    def assign_strata(self, stratify_data: ArrayLike) -> np.ndarray:

        assert self.strata is not None, "Attempting to assign to strata, but strata have not been defined yet!"

        stratum_assignments = np.digitize(stratify_data, self.strata)

        return stratum_assignments

    def fit(self, data: ArrayLike, k: int, stratify_coordinate: ArrayLike, strata: ArrayLike, **kwargs) -> None:

        self.validate_data(data, stratify_coordinate)

        # Assign each stratify_coordinate to a stratum
        # TODO: For now, assuming the strata are ordered
        assert np.equal(strata, np.sort(strata)).all(), "Strata boundaries not sorted!"
        self.strata = strata

        stratum_assignments = self.assign_strata(stratify_coordinate)

        # For each stratum, go through, pick out all the coordinates in it, and cluster on them
        # The total number of strata is the number of provided strata + 2 (the +2 add bins going to -inf and +inf)
        # TODO: Handle if the user already has bounds going from -inf to +inf
        self.strata_kmeans = [
            KMeans(n_clusters=k, **kwargs)
            for _ in range(self.n_strata)
        ]

        for stratum_index in range(self.n_strata):
            # TODO: Handle if you don't have enough points in this bin! What do we do then? Probably just error.

            stratum_kmeans = self.strata_kmeans[stratum_index]

            points_in_stratum = np.where(stratum_assignments == stratum_index)[0]
            data_in_stratum = data[points_in_stratum]

            if len(points_in_stratum) == 0:
                print(f"No points in stratum {stratum_index}!")
                continue

            stratum_kmeans.fit(data_in_stratum)

    def predict(self, data: ArrayLike, stratify_coordinate: ArrayLike) -> np.ndarray:

        self.validate_data(data, stratify_coordinate)

        assert self.strata_kmeans is not None, "Attempting to predict, but stratum KMeans models have not been fit()!"

        stratum_assignments = self.assign_strata(stratify_coordinate)

        # Each stratum will assign cluster IDs that are indexed from 0, within that stratum.
        # In order to get globally unique cluster numbers, we'll just keep a counter. So, if we have 5 clusters in
        #   stratum 0, then the 0th cluster in stratum 1 will be assigned id 5.
        stratum_cluster_offset = 0

        # We can either have one long trajectory, or many N-dimensional trajectories
        # So if data has only 1 dimension, then we have n_traj=1, length=len(data)
        # TODO: There's probably a smarter way to do this with a reshape.

        if len(data.shape) == 1:
            n_traj = 1
            traj_len = len(data)
        else:
            n_traj = data.shape[0]
            traj_len = data.shape[1]

        cluster_assignments = np.zeros((n_traj, traj_len), dtype=int)

        for stratum_index in range(self.n_strata):

            stratum_kmeans = self.strata_kmeans[stratum_index]

            points_in_stratum = np.where(stratum_assignments == stratum_index)[0]
            data_in_stratum = data[points_in_stratum]

            if len(data_in_stratum) == 0:
                continue  # Not predicting any datapoints in this stratum

            points = stratum_kmeans.predict(data_in_stratum) + stratum_cluster_offset
            points = points.reshape(data_in_stratum.shape)
            cluster_assignments[points_in_stratum] = points

            stratum_cluster_offset += len(stratum_kmeans.cluster_centers_)

        assert len(cluster_assignments) == len(data), "Sanity check on clustering failed! Number of cluster " \
                                                      "assignments doesn't match input data."

        return cluster_assignments
