# fond

## 1) Testing
```bash
#!/bin/bash

DATA_DIR=/pub2/data/ python -m unittest domainbed.test.test_datasets.TestOverlapDatasets

```

## 2) Downloading Datasets
```bash
#!/bin/bash

python -m src.utils.download --datadir=/pub2/data

```
