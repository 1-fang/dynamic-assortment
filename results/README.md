# Manuscript results

**Compiling Operational Knowledge for Dynamic Assortment: LLM-Guided Policy Learning under Resource Constraints**

Numbering and reported values follow the RQ1–RQ3 manuscript (`main_1113.tex`). The [reproduction notebook](../reproduce_results.ipynb) reads these CSVs. The [machine-readable index](index.csv) lists each file and its source.

- Reported means and standard deviations occupy separate columns and keep the manuscript's displayed precision. The CSVs repeat group labels where the typeset tables use continuation rows.
- Expedia30 reports daily booking revenue. The other Expedia settings report booking revenue per search. KuaiSearch reports expected GMV per 1,000 requests, with prices scaled by 1/100.
- Table EC.15 reports frozen deployment. Tables EC.17 and EC.18 report policies trained separately at each capacity. Figures use the menu specified by their source tables.
- Figure 5 evaluates the manuscript's illustrative inventory cost and daily S/R charge. Figure EC.3 uses the least tested capacity attaining each target. Its interval rows represent `(target_left, target_right]`. Derived percentages and breakpoints are exported to eight decimals for plotting.
- Figures 1 and 2 reference the conceptual illustrations. Figure EC.1 has a header-only CSV because the manuscript contains the curve image without the numerical epoch series.

## Main tables

| Exhibit | Contents | Source |
| --- | --- | --- |
| [Table 1](table-01.csv) | Structural comparison of the two dynamic assortment regimes | Reported in manuscript |
| [Table 2](table-02.csv) | Empirical populations, temporal data splits, and demand-support scenarios across the nine operating settings | Reported in manuscript |
| [Table 3](table-03.csv) | Out-of-sample operating performance across booking-depleted hotel inventory and display-depleted product exposure | Reported in manuscript |
| [Table 4](table-04.csv) | Channel ablations and representation-source comparisons on A2C Transformer across all nine settings | Reported in manuscript |
| [Table 5](table-05.csv) | Frozen-policy deployment synthesis: Topline expansion vs. premium dilution and peak-target capacity substitution | Reported in manuscript |

## Main figures

| Exhibit | Contents | Source |
| --- | --- | --- |
| [Figure 1](figure-01.csv) | Sequential assortment control under dual resource-depletion regimes | Conceptual figure |
| [Figure 2](figure-02.csv) | Offline LLM compilation and sequential deployment framework | Conceptual figure |
| [Figure 3](figure-03.csv) | Share of Full S/R gain captured by the stronger single interface | Table EC.9 |
| [Figure 4](figure-04.csv) | Retained frozen-policy uplift ranges across 41 capacity conditions | Tables EC.15 and EC.16 |
| [Figure 5](figure-05.csv) | Joint policy and capacity selection in Expedia30 | Derived from Table EC.17 and the manuscript cost parameterization |

## Appendix tables

| Exhibit | Contents | Source |
| --- | --- | --- |
| [Table EC.1](table-ec-01.csv) | Pre-decision input schema and non-anticipative boundaries for executable S/R programs | Reported in manuscript |
| [Table EC.2](table-ec-02.csv) | Summary of compiled state (S^*) and reward (R^*) interface dimensions | Reported in manuscript |
| [Table EC.3](table-ec-03.csv) | Compiled Global State Builder (S^*) features for Expedia30 | Reported in manuscript |
| [Table EC.4](table-ec-04.csv) | Pre-decision base global summaries (g_t,1:8^0) and item primitives supplied to heterogeneous-hotel compilation | Reported in manuscript |
| [Table EC.5](table-ec-05.csv) | Tier-specific global state recombinations (f_t^global) in validated heterogeneous-hotel S^* programs | Reported in manuscript |
| [Table EC.6](table-ec-06.csv) | Compiled Global State Builder (S^*) features for KuaiSearch Overall | Reported in manuscript |
| [Table EC.7](table-ec-07.csv) | Consolidated comparison of compiled S^* features across KuaiSearch query-support tiers (Q-Head, Q-Regular, Q-Cold, and Q-Unseen) | Reported in manuscript |
| [Table EC.8](table-ec-08.csv) | Validated Local Reward Reshaper (R^*) formulations across KuaiSearch environments | Reported in manuscript |
| [Table EC.9](table-ec-09.csv) | Channel capture ratios and normalized S/R interaction contrasts across all nine operating settings | Reported in manuscript |
| [Table EC.10](table-ec-10.csv) | Optimizer portability: Out-of-sample performance of whole-slate PPO with and without Qwen-compiled S/R | Reported in manuscript |
| [Table EC.11](table-ec-11.csv) | Pretrained semantic text-encoder specifications and native embedding dimensions | Reported in manuscript |
| [Table EC.12](table-ec-12.csv) | Out-of-sample performance of frozen pretrained text encoders with the A2C GRU backbone | Reported in manuscript |
| [Table EC.13](table-ec-13.csv) | Out-of-sample performance of frozen pretrained text encoders with the A2C RNN-Pointer backbone | Reported in manuscript |
| [Table EC.14](table-ec-14.csv) | Complete comparison of frozen pretrained text encoders against Base and Qwen-compiled S/R on A2C Transformer | Reported in manuscript |
| [Table EC.15](table-ec-15.csv) | Complete zero-shot frozen-policy performance across all 41 resource conditions (no retraining across target capacities) | Reported in manuscript |
| [Table EC.16](table-ec-16.csv) | Summary statistics of relative uplift g_c(r) across the 41-condition frozen-policy grid | Reported in manuscript |
| [Table EC.17](table-ec-17.csv) | Condition-specific hotel inventory grid: Base (A2C Transformer) and Qwen-compiled S/R trained and evaluated at each stock level q | Reported in manuscript |
| [Table EC.18](table-ec-18.csv) | Condition-specific product-exposure grid: Base (A2C Transformer) and Qwen-compiled S/R trained and evaluated at each exposure multiplier ρ (expected GMV per 1,000 requests; prices scaled by 1/100) | Reported in manuscript |

## Appendix figures

| Exhibit | Contents | Source |
| --- | --- | --- |
| [Figure EC.1](figure-ec-01.csv) | Epoch-wise validation-revenue trajectories | Manuscript figure; numerical epoch series unavailable in the manuscript |
| [Figure EC.2](figure-ec-02.csv) | Deployment map: reference-capacity uplift and resource sensitivity | Tables EC.15 and EC.16 |
| [Figure EC.3](figure-ec-03.csv) | Minimum tested capacity required for a revenue target | Derived from Tables EC.17 and EC.18 |
| [Figure EC.4](figure-ec-04.csv) | Qwen S/R gain over the unaugmented A2C Transformer | Derived from Table 3 |
| [Figure EC.5](figure-ec-05.csv) | Hotel market tier revenue and comparator-dependent Qwen S/R gain | Derived from Table 3 |
| [Figure EC.6](figure-ec-06.csv) | Relative revenue gains for four LLM compilers | Derived from Table 3 |
| [Figure EC.7](figure-ec-07.csv) | Text encoder and compiled S/R gains versus A2C Transformer | Derived from Table EC.14 |
| [Figure EC.8](figure-ec-08.csv) | Whole-slate PPO gain from adding Qwen S/R | Derived from Table EC.10 |
| [Figure EC.9](figure-ec-09.csv) | Condition-specific hotel inventory response | Table EC.17 |
| [Figure EC.10](figure-ec-10.csv) | Condition-specific exposure response for query-support groups | Table EC.18 |

## Appendix prompt parameter guide

| Exhibit | Contents | Source |
| --- | --- | --- |
| [Box EC.1 parameter guide](box-ec-01-parameter-guide.csv) | Template parameter guide and heterogeneous-hotel instantiation | Reported in manuscript |
