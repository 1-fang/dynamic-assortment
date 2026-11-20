# Selected experiment implementation

These modules expose the principal computational steps used in the assortment experiments. Data preparation, fitted response artifacts, generated-program search, selected S/R source, training checkpoints, and orchestration are outside the release scope. The experiment modules illustrate the implementation, while the exhibit scripts reproduce the manuscript aggregates directly from the supplied data.

| Module | Role |
| --- | --- |
| [Hotel policy](hotel_policy.py) | Set encoder, state adapter, masked pointer decoding and STOP action |
| [Product policy](product_policy.py) and [catalog encoder](catalog_encoder.py) | Candidate and query encoding with budget-aware assortment selection |
| [Training](training.py) | A2C actor and critic loss with generalized advantage estimation |
| [Response](response.py) | Sampled booking revenue and expected product revenue using an outside option |
| [Exposure](exposure.py) | Weighted per-item allocation with occurrence caps and per-display accounting |
| [Metrics](metrics.py) | Average daily revenue, revenue per search, and expected GMV per 1,000 requests |

The hotel policy accepts an S program via a `features` interface. It returns eight global features and four features per candidate for the variable-candidate hotel setting. The fixed 30-hotel setting uses eight global features and three per hotel. When an R program is used, its output is added to the training reward with the scheduled coefficient. The outcome reported in each table uses business reward. The configured response and policy choices can vary by setting.

Run a CPU check using constructed inputs from the repository root:

```bash
python -m pip install -r experiment/requirements.txt
python -m experiment.demo
```

The example checks masks, empty slates, the A2C loss, expected revenue and the exposure budget using a small constructed batch. Table values come from the manuscript exhibit files.
