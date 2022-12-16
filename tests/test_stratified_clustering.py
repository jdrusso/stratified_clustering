import unittest
import numpy as np
from stratified_clustering.stratified_clustering import StratifiedClusterer


class TestStratifiedClustering(unittest.TestCase):

    def setUp(self) -> None:
        self.sample_bounds = np.array([0, 1, 2, 3, 4, 5])

        self.sample_stratify_data = np.array([-5, -3,  # A few numbers below the bottom bounds
                                              1.5, 1.25, 2.5, 2.3,  # A few clean ones
                                              10, 11, 12  # One that's above the upper bound
                                              ]).reshape(-1, 1)

        self.k = 2
        self.seed = 1337

    def test_identical_fit_and_predict(self) -> None:
        # This is just a bunch of points that are all 1 -- this will be used to show that stratified clustering
        #   gives different cluster IDs for the same point in a different stratum.
        self.identical_sample_data = np.ones_like(self.sample_stratify_data)
        self.identical_reference_assignments = np.array([0, 0, 2, 2, 4, 4, 6, 6, 6]).reshape(-1, 1)

        clusterer = StratifiedClusterer()

        clusterer.fit(
            data=self.identical_sample_data,
            k=self.k,
            stratify_coordinate=self.sample_stratify_data,
            strata=self.sample_bounds,
            random_state=self.seed
        )

        assignments = clusterer.predict(self.identical_sample_data, self.sample_stratify_data)

        assert np.all(assignments == self.identical_reference_assignments)

    def test_fit_and_predict(self) -> None:
        self.sample_data = np.array([
            1, 5,
            1.1, 1.2, 3, 1,
            2, 3, 1
        ]).reshape(-1, 1)
        self.reference_assignments = np.array([
            1, 0,
            3, 2, 5, 4,
            7, 6, 7
        ]).reshape(-1, 1)

        clusterer = StratifiedClusterer()

        clusterer.fit(
            data=self.sample_data,
            k=self.k,
            stratify_coordinate=self.sample_stratify_data,
            strata=self.sample_bounds,
            random_state=self.seed
        )

        assignments = clusterer.predict(self.sample_data, self.sample_stratify_data)

        assert np.all(assignments == self.reference_assignments), assignments


if __name__ == '__main__':
    unittest.main()
