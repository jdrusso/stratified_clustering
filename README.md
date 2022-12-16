# Stratified Clustering


[//]: # ([![Pypi link]&#40;https://img.shields.io/pypi/v/stratified_clustering.svg&#41;]&#40;https://pypi.python.org/pypi/stratified_clustering&#41;)

[//]: # ([![Travis job]&#40;https://img.shields.io/travis/russojd/stratified_clustering.svg&#41;]&#40;https://travis-ci.org/russojd/stratified_clustering&#41;)




Implementation of stratified clustering 

[//]: # (## Table of Content:)

[//]: # ()
[//]: # (- [Intallation]&#40;#installation&#41;)

[//]: # (- [Usage]&#40;#usage&#41;)

[//]: # (- [TODO]&#40;#todo&#41;)

[//]: # (- [Contributing]&#40;#contributing&#41;)

[//]: # (- [Credits]&#40;#credits&#41;)

[//]: # ()
[//]: # (## Installation)

[//]: # ()
[//]: # ()
[//]: # (```batch)

[//]: # ()
[//]: # (    $ pip install stratified_clustering)

[//]: # (```)

[//]: # ()
[//]: # (This is the preferred method to install Stratified Clustering, as it will always)

[//]: # (install the most recent stable release.)

[//]: # ()
[//]: # (If you don't have [pip]&#40;https://pip.pypa.io&#41; installed, this )

[//]: # ([Python installation guide]&#40;http://docs.python-guide.org/en/latest/starting/installation/&#41; )

[//]: # (can guide you through the process.)

## Usage

```python
import numpy as np
from stratified_clustering.stratified_clustering import StratifiedClusterer

# Define the data/coordinate we'll stratify on
stratification_data = np.array([1.1, 2.3, 3.1, 4.7])

# Define the bounds of our strata
# This will group the samples by stratification coordinates -- the groups will be (1.1), (2.3, 3.1), (4.7)
strata_boundaries = np.array([1.5, 3.5])

# Now define the data you actually want to cluster on. Here, we'll have 4x 3-dimensional samples, like 4 frames of 3D coords.
# Remember -- the clustering only operates on these coordinates, but will be done independently for the data in each stratum.
data = np.array([
    [0.5, 1.2, 0.1],  # In group 1, because it has stratification coordinate 1.1
    [1.3, 6.3, 1.1],  # In group 2, stratification coordinate is 2.3
    [4.7, 2.3, 1.6],  # In group 2, stratification coordinate is 3.1
    [1.6, 5.3, 8.1],  # In group 3, stratification coordinate is 4.7
])

clusterer = StratifiedClusterer()

n_clusters = 5
clusterer.fit(
    data = data,
    k = n_clusters,
    stratify_coordinate = stratification_data,
    strata = strata_boundaries
)

test_stratification_data = np.array([
    2, 
    4
])
test_data = np.array([
    [1.2, 1.4, 9.8],
    [5.5, 2.3, 8.2]
])

cluster_assignments = clusterer.predict(
    data = test_data, 
    stratify_coordinate = test_stratification_data
)

```


[//]: # (## TODO)

[//]: # ()
[//]: # (- [ ] Add Test)


[//]: # (## Contributing)

[//]: # ()
[//]: # (Contributions are welcome, and they are greatly appreciated! Every)

[//]: # (little bit helps, and credit will always be given.)

[//]: # ()
[//]: # (For more info please click [here]&#40;./CONTRIBUTING.md&#41;)


## Credits

This package was created with Cookiecutter and the `oldani/cookiecutter-simple-pypackage` project template.

- [Cookiecutter](https://github.com/audreyr/cookiecutter)
- [oldani/cookiecutter-simple-pypackage](https://github.com/oldani/cookiecutter-simple-pypackage)
