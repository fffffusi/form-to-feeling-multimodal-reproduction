# Product visual form and customer satisfaction

Public files associated with the manuscript:

> **Product visual form and customer satisfaction: multimodal evidence on affective value and functional diagnosis**

This repository provides anonymized product-level data together with the analysis code, validation materials, dictionaries, prompts, visual summaries, and supplementary diagnostic scripts associated with the study.

## Repository structure

```text
form-to-feeling-multimodal-reproduction/
├── README.md
├── requirements.txt
│   └── Python dependencies
│
├── data_anonymized/
│   ├── product_master.csv
│   │   └── Anonymized product-level master data
│   └── product_master.dta
│       └── Stata version of the anonymized master data
│
├── data_validation/
│   ├── extraction_coding_LLM.xlsx
│   │   └── LLM coding results for construct validation
│   ├── extraction_coding_coder1.csv
│   │   └── Human coder 1 validation data
│   ├── extraction_coding_coder2.csv
│   │   └── Human coder 2 validation data
│   └── sentiment_coding_validation.xlsx
│       └── Human validation data for sentiment scoring
│
├── diagnostics/
│   ├── cmb_harman_single_factor.csv
│   │   └── Common-method diagnostic output
│   └── run_mf_complete_robustness.py
│       └── Functional-evaluation decomposition and robustness analysis
│
├── lexicons/
│   ├── kansei_dict.json
│   │   └── Affective/Kansei dictionary
│   ├── function_dict.json
│   │   └── Functional-evaluation dictionary
│   ├── degree_words.json
│   │   └── Degree-word dictionary
│   ├── negation_words.json
│   │   └── Negation-word dictionary
│   ├── kansei_extraction.txt
│   │   └── LLM prompt for affective-evaluation extraction
│   ├── function_extraction.txt
│   │   └── LLM prompt for functional-evaluation extraction
│   └── sentiment_score.txt
│       └── LLM prompt for sentiment scoring
│
├── scripts_extraction/
│   ├── llm_extract_kansei.py
│   │   └── Extracts affective evaluations from reviews
│   ├── llm_extract_function.py
│   │   └── Extracts functional evaluations from reviews
│   ├── llm_sentiment_scoring.py
│   │   └── Scores review-level sentiment
│   └── compute_kansei_function_scores.py
│       └── Aggregates review-level evaluations to product-level scores
│
├── scripts_satisfaction/
│   └── compute_product_satisfaction.py
│       └── Constructs product-level satisfaction measures
│
├── scripts_visual/
│   ├── encode_clip_images.py
│   │   └── Extracts CLIP image embeddings
│   ├── encode_clip_text.py
│   │   └── Computes image-text concept scores
│   ├── produce_onehot.xlsx
│   │   └── Engineering one-hot visual coding
│   ├── fig7(a)_product_visual_pcs.csv
│   │   └── Product-level visual principal-component scores
│   └── fig7(b)_product_feature_contributions.csv
│       └── Visual feature contributions and loading information
│
├── scripts_stata_prep/
│   ├── prepare_stata_data.py
│   │   └── Prepares product-level data for Stata
│   ├── fdr_correction.py
│   │   └── Benjamini-Hochberg FDR correction for the 12-mediator decomposition
│   ├── joint_significance_fdr.py
│   │   └── Joint-significance and FDR analysis for 14 PC-specific indirect pathways
│   └── common_method_and_diagnostic_checks.py
│       └── Common-method and functional-evaluation diagnostics
│
└── stata/
    ├── main_analysis.do
    │   └── Main path regressions, BCa-bootstrap indirect associations, robustness checks, and diagnostics
    ├── main_analysis.log
    │   └── Stata execution log
    ├── robustness_analysis.do
    │   └── Supplementary multiple-testing and review-volume sensitivity analyses
    └── robustness_analysis.log
        └── Concise public audit log without machine paths, timestamps, or repeated tables
```

## Main analysis command

The main Stata script reads:

```text
data_anonymized/product_master.dta
```

Run it from the repository root:

```stata
do stata/main_analysis.do
```

The script writes analysis tables and diagnostic files to:

```text
stata/output/
```

Bootstrap output reports both percentile and bias-corrected and accelerated (BCa) confidence intervals.

## Robustness analyses

The supplementary Stata script addresses two additional checks:

1. fourteen component-specific indirect pathways formed by seven visual principal components and two evaluation channels;
2. sensitivity of the platform favorable-rate regressions to review-volume precision, using minimum-review thresholds of 50 and 100 and a proxy review-volume weighted specification.

Run the supplementary Stata analysis from the repository root:

```stata
do stata/robustness_analysis.do
```

This script reproduces the required transformations from the anonymized public data, performs 5,000 product-level bootstrap replications, and writes locally generated CSV, DTA, bootstrap-log, and RTF table files to:

```text
stata/output_robustness/
```

The generated output directory is intentionally not versioned. The repository includes only `stata/robustness_analysis.log`, a concise audit record without machine-specific paths, execution timestamps, or duplicated regression tables.

After the Stata script completes, run:

```bash
python scripts_stata_prep/joint_significance_fdr.py
```

For each indirect pathway, the script tests the composite null hypothesis
`H0: a*b = 0` using the joint-significance p-value `max(p_a, p_b)`. It then
applies Benjamini-Hochberg (BH) and Benjamini-Yekutieli (BY) adjustments across
the family of 14 pathway-level p-values. Indirect-association magnitudes,
bootstrap standard errors, and BCa confidence intervals remain those generated
by the 5,000-replication Stata bootstrap.

The existing `scripts_stata_prep/fdr_correction.py` serves a different purpose:
it adjusts the 12 p-values from the affective and functional subdimension
decomposition. It should not be used in place of the 14-path joint-significance
analysis.

`n_reviews_total` is the number of review texts collected for each product. It
has not been verified as the exact denominator used by the platform to calculate
the favorable rate. The weighted specification is therefore a proxy sensitivity
analysis, whereas the minimum-review-threshold analyses provide the more direct
sample-restriction checks.

The analysis uses Stata 18 and the `estout` package:

```stata
ssc install estout, replace
```

## Python environment

The Python scripts were developed with Python 3.10–3.12.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

The LLM extraction scripts require an OpenAI-compatible API key. The CLIP scripts require the corresponding model weights and a suitable PyTorch installation. Raw review text, product images, private model files, and platform-identifying materials are not included because of data-access, brand, and platform-risk considerations.

## Main product-level variables

The anonymized master data include:

- platform favorable-rate `y_favrate_only` and alternative satisfaction measures;
- aggregate affective evaluation `m_e_avg`;
- aggregate functional evaluation `m_f_avg`;
- affective subdimensions `me1_z`–`me5_z`;
- functional subdimensions `mf1_z`–`mf7_z`;
- visual components `z_pc1`–`z_pc7`;
- product price and review count, with log transformations generated by the Stata script;
- discount and free-shipping indicators;
- seller-reputation gap;
- log brand-average price;
- review recency.

## References

- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*, 57(1), 289–300.
- Benjamini, Y., & Yekutieli, D. (2001). The control of the false discovery rate in multiple testing under dependency. *The Annals of Statistics*, 29(4), 1165–1188. https://doi.org/10.1214/aos/1013699998
- MacKinnon, D. P., Lockwood, C. M., Hoffman, J. M., West, S. G., & Sheets, V. (2002). A comparison of methods to test mediation and other intervening variable effects. *Psychological Methods*, 7(1), 83–104. https://doi.org/10.1037/1082-989X.7.1.83
- Radford, A., Kim, J. W., Hallacy, C., et al. (2021). Learning transferable visual models from natural language supervision. *Proceedings of the 38th International Conference on Machine Learning*.
- Yzerbyt, V., Muller, D., Batailler, C., & Judd, C. M. (2018). New recommendations for testing indirect effects in mediational models: The need to report and test component paths. *Journal of Personality and Social Psychology*, 115(6), 929–943. https://doi.org/10.1037/pspa0000132

Please cite the associated manuscript when using these files.
