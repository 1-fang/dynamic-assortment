# Aggregate exhibit data

Each numbered table has a JSON file in `tables/`. The `panels` array preserves row and panel order. A numeric result cell contains `mean` and `sd` as decimal strings at the manuscript's displayed precision. Empty group cells inherit the preceding group. Text cells retain headings, scenario names and units.

`current_results.json` presents the plot inputs in a long format. `scripts/check_figure_data.py` compares its mean and SD pairs with Tables 3, 4, 6, 7, and 8. Table EC.16 supplies the frozen migration grid for the deployment map. Its results describe frozen-policy evaluation across target resource conditions. In Tables 7 and 8 each resource condition uses a policy trained for that resource level.

Expedia30 uses revenue per day. The three Expedia market groups use revenue per search. KuaiSearch uses expected GMV per 1,000 requests, with prices divided by 100 in the evaluation. The Overall and query-group evaluations use separate sequences and budgets. Reported SDs describe dispersion, and the scripts compute changes from the displayed means.
