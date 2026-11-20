<h1 align="center">Large Language Models for Dynamic Assortment</h1>
<p align="center"><strong>A Knowledge-Guided Learning Approach</strong></p>
<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#data">Data</a> ·
  <a href="EXHIBITS.md">Tables and figures</a> ·
  <a href="experiment/README.md">Experiment code</a>
</p>

## Overview

Dynamic assortment policies spend scarce resources across successive customer requests. Expedia hotel inventory is consumed on booking. KuaiSearch exposure capacity is consumed on display. In both settings, a decision changes the resources available to serve later requests.

<details open>
<summary><strong>Background · Two resource environments</strong></summary>
<br>
<p align="center"><img src="assets/background.png" width="950" alt="Side-by-side comparison of Expedia inventory and KuaiSearch exposure constraints"></p>
</details>

The framework uses an LLM during offline development to propose executable programs. The **Global State Builder (S)** enriches the policy input with relationships among demand, remaining resources, and available candidates. The **Local Reward Reshaper (R)** provides auxiliary feedback during training. At inference, the trained policy uses S and a feasibility mask to select the assortment.

<details open>
<summary><strong>Method · Program design, learning, and deployment</strong></summary>
<br>
<p align="center"><img src="assets/method.png" width="950" alt="Four-stage framework for task inputs, S/R synthesis, validation and learning, and constrained deployment"></p>
</details>

## Data

The experiments cover nine settings built from Expedia and KuaiSearch-Lite. The data guide lists the dataset sources, train, validation and test counts, market and query tiers, and the revenue metrics. [Read the data guide](data/README.md)

| Directory | Contents |
| --- | --- |
| [`assets/`](assets/) | Background and method illustrations |
| [`data/`](data/) | Manuscript aggregate statistics and table cells in JSON |
| [`scripts/`](scripts/) | Table rendering, plot checks, and figure generation |
| [`experiment/`](experiment/) | Selected implementation modules and a CPU example |

The release includes aggregate data for eight main tables and sixteen appendix tables, plus the S/R parameter guide. It includes selected implementation code for feasible assortment selection, A2C updates, customer-response evaluation, and resource accounting. Fig. 3 includes a plotting script with a documented input format. Its epoch-level inputs are outside the current release.

The [MIT License](LICENSE) applies to the released software and its documentation. The figure assets and result summaries accompany this research artifact. Access to the underlying platform interaction datasets follows their respective provider terms.
