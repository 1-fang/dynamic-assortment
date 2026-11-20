# Manuscript exhibit index

See the [current CSV exhibit index](results/README.md) and [reproduction notebook](reproduce_results.ipynb) for the RQ1–RQ3 manuscript.

## Earlier JSON release

The entries below follow the numbering used by the earlier JSON release. Each table can be rendered with [`render_tables.py`](scripts/render_tables.py). The script checks the reported differences in Tables 7, 8, and EC.16 from the displayed means.

## Tables

| Number | Exhibit | Inputs |
| --- | --- | --- |
| 1 | Resource environments | [JSON](data/tables/table-1.json) |
| 2 | Data populations and splits | [JSON](data/tables/table-2.json) |
| 3 | Main policy performance | [JSON](data/tables/table-3.json) |
| 4 | Embedding and S/R comparison | [JSON](data/tables/table-4.json) |
| 5 | S/R component ablations | [JSON](data/tables/table-5.json) |
| 6 | PPO comparison | [JSON](data/tables/table-6.json) |
| 7 | Hotel inventory grid | [JSON](data/tables/table-7.json) |
| 8 | Product exposure grid | [JSON](data/tables/table-8.json) |
| EC.1 | Executable inputs | [JSON](data/tables/table-ec-1.json) |
| EC.2 | Output dimensions | [JSON](data/tables/table-ec-2.json) |
| EC.3 | Expedia30 S dimensions | [JSON](data/tables/table-ec-3.json) |
| EC.4 | Hotel operating summaries | [JSON](data/tables/table-ec-4.json) |
| EC.5 | Hotel global features | [JSON](data/tables/table-ec-5.json) |
| EC.6 | Overall S dimensions | [JSON](data/tables/table-ec-6.json) |
| EC.7 | Q-Head S dimensions | [JSON](data/tables/table-ec-7.json) |
| EC.8 | Q-Regular S dimensions | [JSON](data/tables/table-ec-8.json) |
| EC.9 | Q-Cold S dimensions | [JSON](data/tables/table-ec-9.json) |
| EC.10 | Q-Unseen S dimensions | [JSON](data/tables/table-ec-10.json) |
| EC.11 | Product-search R feedback | [JSON](data/tables/table-ec-11.json) |
| EC.12 | Embedding treatments | [JSON](data/tables/table-ec-12.json) |
| EC.13 | GRU embedding results | [JSON](data/tables/table-ec-13.json) |
| EC.14 | RNN-Pointer embedding results | [JSON](data/tables/table-ec-14.json) |
| EC.15 | Transformer-Pointer embedding results | [JSON](data/tables/table-ec-15.json) |
| EC.16 | Frozen deployment grid | [JSON](data/tables/table-ec-16.json) |
| Box EC.1 guide | Prompt parameter guide | [JSON](data/tables/prompt-parameters.json) |

## Figures

The quantitative figures use the listed aggregate tables via [`current_results.json`](data/current_results.json), except for the deployment map, which reads EC.16 directly. The two supplied diagrams are publication assets.

| Number | Exhibit | Inputs | Plot source |
| --- | --- | --- | --- |
| 1 | Resource environments | [Illustration](assets/background.png) | Author-supplied concept diagram |
| 2 | S/R framework | [Illustration](assets/method.png) | Author-supplied concept diagram |
| 3 | Validation trajectories | CSV input with `setting`, `epoch`, `base_mean`, `sr_mean` columns | [Learning-curve script](scripts/plot_learning_curves.py) |
| 4 | Capacity requirements | [Tables 7](data/tables/table-7.json) and [8](data/tables/table-8.json) | [Planning script](scripts/draw_planning_figures.py) |
| 5 | Capacity and adoption | [Table 7](data/tables/table-7.json), Expedia30 and illustrative inventory charges | [Planning script](scripts/draw_planning_figures.py) |
| EC.1 | Deployment map | [Table EC.16](data/tables/table-ec-16.json) | [Deployment script](scripts/plot_deployment.py) |
| EC.2 | Destination-market support | [Table 3](data/tables/table-3.json), hotel-market comparators | [Summary plots](scripts/plot_updated_results.py) |
| EC.3 | Program-source comparison | [Table 3](data/tables/table-3.json), LLM source rows | [Summary plots](scripts/plot_updated_results.py) |
| EC.4 | Inventory response | [Table 7](data/tables/table-7.json) | [Summary plots](scripts/plot_updated_results.py) |
| EC.5 | Query exposure response | [Table 8](data/tables/table-8.json), Query rows | [Summary plots](scripts/plot_updated_results.py) |
| EC.6 | Main policy gains | [Table 3](data/tables/table-3.json), Qwen versus A2C Transformer | [Summary plots](scripts/plot_updated_results.py) |
| EC.7 | Embedding and executable augmentation | [Tables 3](data/tables/table-3.json) and [4](data/tables/table-4.json) | [Summary plots](scripts/plot_updated_results.py) |
| EC.8 | PPO augmentation | [Table 6](data/tables/table-6.json) | [Summary plots](scripts/plot_updated_results.py) |

The validation-trajectory inputs for Figure 3 are outside the scope of this archive. The renderer accepts verified epoch-wise means in the format above.
The background and framework previews were rasterized from the supplied one-page presentation exports with `pdftoppm -f 1 -l 1 -scale-to 1800 -singlefile -png input.pdf output-stem`. They are conceptual illustrations.
