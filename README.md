[[Paper]](https://doi.org/10.1145/3805689.3812391)

# Conformal Prediction and the Many-Worlds of Fairness

Julia Broden, Jan Simson, and Christoph Kern. 2026. Conformal Prediction and the Many-Worlds of Fairness. In The 2026 ACM Conference on Fairness, Accountability, and Transparency (FAccT ’26), June 25–28, 2026, Montreal, QC, Canada. ACM, New York, NY, USA, 25 pages. https://doi.org/10.1145/3805689.3812391

## Abstract

This paper introduces *Conformal Multiverse Analysis (CMA)*, an auditing framework which integrates Conformal Prediction with Multiverse Analysis. CMA enables a comprehensive assessment of how uncertainty of prediction models propagates across different modeling choices and affects group-conditional coverage of different groups and minorities.

Pairing Multiverse Analysis with Conformal Prediction provides a novel procedure for identifying disparities and critical design decisions, particularly in contexts where uncertainty-aware decision policies are aimed to be deployed.

We present a comprehensive empirical illustration which utilizes large-scale administrative records to build and compare algorithmic profiling models predicting long-term unemployment in two settings. We quantify uncertainty through the *Average Prediction Set Size (APSS)* and miscoverage via the *Group-Conditional Coverage (GCC)* and analyze the resulting outcome distributions using functional ANOVA.
Our results show that (1) design decisions can affect uncertainty outcomes for subgroups and (2) uncertainty reduction and group-specific coverage can be misaligned. Our work reinforces the need and value of integrating uncertainty quantification into fairness-aware model design and evaluation.

## Background

This project is an extension of the [Master's thesis](https://github.com/jsbroden/CMA_Fairness) with the same title, by Julia Sophie Broden, supervised by Christoph Kern.

Both the thesis and Tthis repository build on the code from the paper [One Model Many Scores: Using Multiverse Analysis to Prevent Fairness Hacking and Evaluate the Influence of Model Design Decisions](https://dl.acm.org/doi/10.1145/3630106.3658974) by Jan Simson, Florian Pfisterer and Christoph Kern, published in the proceedings of the ACM Conference on Fairness, Accountability, and Transparency 2024 in Rio de Janeiro, Brazil in June 2024. We adapted their framework to our setting of Conformal Multiverse Analysis (CMA).

## Setup

This project uses [uv](https://docs.astral.sh/uv/). To install dependencies, first [install uv](https://docs.astral.sh/uv/getting-started/installation/), then run `uv sync` in the root directory of the project. Once set up, you can run commands within the virtual environment using `uv run <command>`, or activate it with `source .venv/bin/activate`.

## Organization & Structure

The project is organized into two main analysies, both predicting long-term unemployment (LTU) using conformal prediction. One analysis predicts LTU in binary fashion while the other does so in a more nuanced, multiclass way.

The binary outcome analysis is located in the root of the project, while the multiclass analysis is located in the `multiclass/` directory. The file organization between the two analysis is very similar, however.

- `universe_analysis.ipynb`: The universe analysis scripts describe the universes and actual prediction problems to be analyzed. They are called multiple times with different settings to run the mulvierse analysis.
- `/multiverse_analysis.py` / `multiclass/multiverse.py`: These two files specify the multiverse to be analyzed for the binary and multiclass prediction problems, specifically.
- `output/` / `multiclass/output`: The output directories contain the outputs of the multiverse analyses and the results from analyzing their data.
- `analysis_*.ipynb`: The analysis notebooks contain empirical analysis of the resutls from the multiverse analyses. Analysis notebooks between the binary and multiclass settings are highly similar.
- `/fairness_multiverse/`: This directory contains auxiliary code used during analyses. It also contains the code to orchestrate the binary outcome analysis.

## Adapting to a New Use-Case

The binary-outcome analysis uses custom code to execute the multiverse analysis, whereas the multiclass analysis uses the [`multiversum`](https://simson.io) library to run the multiverse analysis. We therefore recommend the multiclass analysis as a starting point for adapting it to other prediction problems.

# Reproducibility

## Data

This study makes use of the the [Sample of Integrated Labour Market Biographies (SIAB)](https://fdz.iab.de/en/our-data-products/individual-and-household-data/siab/) provided by the [German Institute for Employment Research (IAB)](https://iab.de/en).

We are unfortunately not allowed to share the data used in the analysis due to data protection regulations. However, the data is available upon request from the [German Institute for Employment Research (IAB)](https://iab.de/en) for research purposes.

## Re-running the analysis

The main analysis can be re-run by running the following commands:


```bash
# Run binary analysis
uv run multiverse_analysis.py

# Run multiclass analysis
cd multiclass
uv run -m multiversum
```

Note: The original analysis used the default seed (i.e. the one the libraries default to if no seed is set). However, for train-test splits the seed 42 was used. When running the replication analaysis, we updated the code to use the global seed for the train-test splits as well. If you want to reproduce results from the paper, you will have to update the code to manually use 42 for train-test splits.

## Re-running with different seeds

To check the robustness of results, we re-ran the analysis using different seeds. To replicate this, execute the following commands in this directory.

```bash
uv run multiverse_analysis.py --seed 80539
uv run multiverse_analysis.py --seed 1102
uv run multiverse_analysis.py --seed 47906
uv run multiverse_analysis.py --seed 378
uv run multiverse_analysis.py --seed 68131
```

```bash
cd multiclass
uv run -m multiversum --seed 80539
uv run -m multiversum --seed 1102
uv run -m multiversum --seed 47906
uv run -m multiversum --seed 378
uv run -m multiversum --seed 68131
```

Note: The original analysis used the default seed (i.e. the one the libraries default to if no seed is set). However, for train-test splits the seed 42 was used. Only in the replications is the global seed also used for the train-test split.
