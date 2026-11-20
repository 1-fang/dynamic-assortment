# Data

The study evaluates dynamic assortment decisions in nine operating settings built from Expedia hotel searches and KuaiSearch-Lite product-search sessions. The counts below refer to the study populations after preparation and filtering.

## Expedia

**Expedia30** follows the fixed-catalog benchmark of Li et al. It contains 30 hotels and four customer types. The splits are chronological by day, and each day forms one episode. The policy can offer up to four hotels per request. Reference inventory is two units per hotel, and a booking consumes one unit. [Benchmark paper](https://journals.sagepub.com/doi/10.1177/10591478251351737)

**Expedia destination markets** contain 303,834 retained searches across 2,244 destinations. Searches are ordered within each destination. The target split is 70% training, 10% program validation, 5% model validation, and 15% testing. The table combines the two validation splits and shows the realized counts after filtering. The filters retain positively priced hotels up to the training 99.9th price percentile, require at least five training exposures per retained property, and keep searches with at least two candidates before inventory depletion. An episode stays within one destination and contains at most 120 searches. Reference inventory is one unit per hotel. The Head, Regular, and Cold tiers are based on **destination-level training search counts**.

| Setting | Unit or tier definition | Train | Validation | Test | Total |
| --- | --- | ---: | ---: | ---: | ---: |
| Expedia30 | Days | 140 | 20 | 51 | 211 |
| Expedia30 | Requests within those days | 16,796 | 2,417 | 6,159 | 25,372 |
| Destination markets | All three tiers, requests | 211,714 | 45,488 | 46,632 | 303,834 |
| Head | 206 destinations, 200 or more training searches per destination | 96,793 | 20,752 | 20,856 | 138,401 |
| Regular | 813 destinations, 50–199 training searches per destination | 77,849 | 16,747 | 17,144 | 111,740 |
| Cold | 1,225 destinations, 20–49 training searches per destination | 37,072 | 7,989 | 8,632 | 53,693 |

## KuaiSearch-Lite

The study starts with 555,553 source sessions and retains 384,392 sequential requests across 44 eligible category clusters. Each retained session becomes one request. Products are deduplicated in their first logged order, with up to 100 candidates retained. Requests are ordered chronologically within category clusters and grouped into episodes of up to 200 requests. The policy can display up to 10 products per request. Each display consumes one exposure unit, regardless of purchase. The reference exposure multiplier is 1.00. [Dataset release](https://github.com/benchen4395/KuaiSearch)

**Overall** evaluates mixed-query sequences. The four query tiers construct separate episodes and exposure ledgers by normalized query frequency in the response-model training population. Q-Cold refers to low-frequency queries. For Q-Unseen, the training column shows zero requests in that tier. Its policy uses the shared training pool. The four query tier counts sum to the reported Overall counts. Overall evaluates mixed sequences with their own ledgers, so its results are computed separately.

| Setting | Query support in training | Train | Validation | Test | Total |
| --- | --- | ---: | ---: | ---: | ---: |
| Overall | Mixed queries across 44 clusters | 299,200 | 75,081 | 10,111 | 384,392 |
| Q-Head | 50 or more occurrences | 40,431 | 9,409 | 1,458 | 51,298 |
| Q-Regular | 5–49 occurrences | 73,005 | 14,840 | 2,047 | 89,892 |
| Q-Cold | 1–4 occurrences | 185,764 | 12,681 | 1,574 | 200,019 |
| Q-Unseen | 0 occurrences | 0 | 38,151 | 5,032 | 43,183 |

## Validation and outcome measures

Training data fit the response models and policies and provide the historical statistics. Program validation ranks S/R candidate pairs by business revenue. Model validation selects policy checkpoints. The held-out test sets measure the selected policies using the business outcome alone.

| Setting | Reported outcome | Aggregation |
| --- | --- | --- |
| Expedia30 | Mean daily booking revenue | Sum sampled booking revenue within each test day, then average across the 51 days. |
| Expedia destination markets | Booking revenue per search | Divide total sampled booking revenue by the number of test searches in the tier. |
| KuaiSearch-Lite | Expected GMV per 1,000 requests | Sum price times fitted purchase probability across displayed products and test requests, divide by the number of requests, and multiply by 1,000. Raw item prices are divided by 100. |

All test arrivals count toward the relevant denominator, including arrivals with an empty assortment or zero revenue. The repository provides aggregate results and figures.

## Aggregate exhibit files

The [current exhibit index](../results/README.md) lists manuscript tables and figure inputs in CSV format. The [reproduction notebook](../reproduce_results.ipynb) reads those CSVs.

### Earlier JSON release

Each numbered table has a JSON file in [`tables/`](tables/). The `panels` array preserves row and panel order. A numeric result cell contains `mean` and `sd` as decimal strings at the manuscript's displayed precision. Empty group cells inherit the preceding group. Text cells retain headings, scenario names and units.

[`current_results.json`](current_results.json) presents the plot inputs in a long format. [`check_figure_data.py`](../scripts/check_figure_data.py) compares its mean and SD pairs with Tables 3, 4, 6, 7, and 8. Table EC.16 supplies the frozen migration grid for the deployment map. Its results describe frozen-policy evaluation across target resource conditions. In Tables 7 and 8 each resource condition uses a policy trained for that resource level.

Reported standard deviations describe dispersion across replications. The scripts compute changes from the displayed means. [Earlier exhibit index](../EXHIBITS.md)
