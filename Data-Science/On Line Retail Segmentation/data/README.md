# Data

The raw dataset is **not committed** because of its size (~95 MB).

## Get the data
Download **Online Retail II** from the UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/502/online+retail+ii

Save it as `online_retail_II.csv` in the **project root** (next to `config.py`),
then run the pipeline:

```bash
python run_pipeline.py
```

The pipeline regenerates everything else: `data/processed/` (cleaned parquet,
features, recommendations), `data/powerbi/` (star-schema exports),
`outputs/models/`, `outputs/figures/`, and `outputs/reports/`.

Expected raw schema: `Invoice, StockCode, Description, Quantity, InvoiceDate,
Price, Customer ID, Country` — ~1,067,371 rows.
