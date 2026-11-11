# Public teaching fixtures

These files are intentionally bundled so the demos can run after installation without downloading arbitrary URLs. [PROVENANCE.json](PROVENANCE.json) records the exact byte sizes, SHA-256 hashes, original URLs and attribution. `uv run portfolio bootstrap` recreates missing fixtures from the allowlisted sources.

## NIST AI RMF 1.0

[nist-ai-rmf-1.0.pdf](nist-ai-rmf-1.0.pdf) is the original NIST AI 100-1 publication, January 2023, DOI [10.6028/NIST.AI.100-1](https://doi.org/10.6028/NIST.AI.100-1). Retain the PDF's notices and attribution. Inclusion does not imply NIST endorsement. This is a versioned retrieval corpus, not a claim that 1.0 is the newest guidance.

## Iris

[iris.csv](iris.csv) is converted from UCI's `iris.data`, with descriptive headers added and blank rows removed. Attribution: Fisher, R. A. (1936), Iris, UCI Machine Learning Repository, DOI [10.24432/C56C76](https://doi.org/10.24432/C56C76). The repository distributes it under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The CSV conversion is the only intended modification; species labels are retained. Exact duplicates present in the dataset are reported by the data profiler.

## Diabetes

[diabetes.csv](diabetes.csv) is exported from scikit-learn's bundled `load_diabetes(as_frame=True)` with its default scaled features. The target column is renamed `disease_progression`. Attribution: Efron et al. (2004), Least Angle Regression; scikit-learn dataset maintainers. See the [official dataset documentation](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) and bundled [description](diabetes-DESCRIPTION.txt). This educational regression experiment is not a medical application.
