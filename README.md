# Product visual form and customer satisfaction

Public files associated with the manuscript:

> **Product visual form and customer satisfaction: multimodal evidence on affective value and functional diagnosis**

This repository contains the anonymized product-level data, analysis code, validation materials, dictionaries, prompts, visual outputs, and supplementary diagnostic scripts used in the study.

## Repository structure

```text
AAA公开public/
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
│   │   └── Benjamini-Hochberg FDR correction
│   └── common_method_and_diagnostic_checks.py
│       └── Common-method and functional-evaluation diagnostics
│
└── stata/
    ├── main_analysis.do
    │   └── Main regressions, indirect associations, robustness checks, and diagnostics
    └── main_analysis.log
        └── Stata execution log
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

- platform favorable-rate and alternative satisfaction measures;
- aggregate affective evaluation `ME`;
- aggregate functional evaluation `MF`;
- affective subdimensions `ME1`–`ME5`;
- functional subdimensions `MF1`–`MF7`;
- visual components `PC1`–`PC7`;
- log price and log review count;
- discount and free-shipping indicators;
- seller-reputation gap;
- log brand-average price;
- review recency.

## References

- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*, 57(1), 289–300.
- Radford, A., Kim, J. W., Hallacy, C., et al. (2021). Learning transferable visual models from natural language supervision. *Proceedings of the 38th International Conference on Machine Learning*.

Please cite the associated manuscript when using these files.
