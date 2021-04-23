##

Implementation of a metric learning method for subspace-valued data.

[1] "",.

## Usage

```python

from subs_ml.models import AbasedMetricLearningSubspace as AMLS

# Prepair data
model = AMLS(
    subspace_dimension,
    n_neighbors=3,
    normalize=True,
    verbose=0,
    max_iter=200,
    tol=1e-8,
    weight_trnorm=0,
    min_iter=5,
)

model.fit(x_train, y_train)

pred = model.predict(x_test)
```