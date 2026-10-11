# Data

The input data are not tracked in this repository. Put these two files in this folder:

| File | Format |
|---|---|
| `expression_matrix.csv` | Samples × genes. The first column is the sample ID and the other columns are gene symbols. Values are log2-scale (range about 1.8–14.4). |
| `labels.csv` | Columns `SampleID` and `Label`, where 1 = HCC and 0 = non-tumor liver. |

## Source

The data come from **GEO [GSE62232](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE62232)**, "Large-scale gene expression profiling of 81 hepatocellular carcinomas", on the Affymetrix HG-U133 Plus 2.0 platform (GPL570).

The analysis uses 91 arrays (GSM1523329–GSM1523419):

- **81 HCC arrays (label 1)**, each from a different patient.
- **10 non-tumor liver arrays (label 0)**, from only **5 patients**:

| Patient | Arrays |
|---|---|
| 0591N | 3 |
| 0934N | 3 |
| 0932N | 2 |
| 0469N | 1 |
| 0562N | 1 |

Patient 0469 has both a tumor and a non-tumor sample.

The patient ID is in the sample name (`JZN_<id><N|T>`). Use `--group-pattern 'JZN_(\d+[NT])'` so that replicate arrays from one patient never end up in both train and test.

The steps used to turn CEL files into gene-level values (normalisation and probe-to-gene collapsing) are not recorded here. Add them to make the analysis fully reproducible.
