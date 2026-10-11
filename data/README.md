# Data

Raw data are not tracked in this repository. Place the following files here:

| File | Format |
|---|---|
| `expression_matrix.csv` | Samples × genes. First column = sample ID; remaining columns = gene symbols (suffixes such as `GENE.1` are stripped). |
| `labels.csv` | First column = sample ID; one column `Label` with binary class (0 / 1). |

Sample IDs are upper-cased and stripped before the two files are aligned, so only samples present in both are used.

The analysis in `results/` used 91 aligned samples (Affymetrix HG-U133 Plus 2.0 arrays from GEO; 10 class-0, 81 class-1) and 21,358 genes.
