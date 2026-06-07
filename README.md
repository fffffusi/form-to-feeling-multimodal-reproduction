# From Form to Feeling: Public Reproduction Materials

This repository contains the public reproduction materials for the paper
**From Form to Feeling: Multimodal Evidence on How Product Appearance Shapes
Customer Satisfaction**. The release includes the anonymized product-level
regression dataset, the Stata analysis script, the affective / functional
lexicons and prompts, the LLM extraction scripts, the satisfaction
construction script, the CLIP image / text encoding scripts, and the published
fused-PCA loading matrix.

Raw review text, product images, product-page screenshots, product URLs, user
information, and original brand names are **not** released. Researchers can
either (a) prepare their own data following the schema below and run the full
pipeline, or (b) use `data_anonymized/` and `stata/` directly to reproduce the
regression tables of the paper.

## Repository Structure

```text
.
├── data_anonymized/
│   ├── product_master.csv          # Anonymized product-level master data, N=200
│   └── product_master.dta          # Same data in Stata format
├── lexicons/
│   ├── kansei_dict.json            # 5-dimension affective lexicon
│   ├── function_dict.json          # 7-dimension functional lexicon
│   ├── degree_words.json           # Degree-word weights
│   ├── negation_words.json         # Negation-word rules
│   ├── kansei_extraction.txt       # Affective extraction prompt
│   ├── function_extraction.txt     # Functional extraction prompt
│   └── sentiment_score.txt         # Sentiment scoring prompt
├── scripts_extraction/
│   ├── llm_sentiment_scoring.py
│   ├── llm_extract_kansei.py
│   ├── llm_extract_function.py
│   └── compute_kansei_function_scores.py
├── scripts_satisfaction/
│   └── compute_product_satisfaction.py
├── scripts_visual/
│   ├── encode_clip_images.py
│   ├── encode_clip_text.py
│   ├── produce_onehot.xlsx                       # Engineering one-hot coding sheet
│   ├── fig7(a)_product_visual_pcs.csv            # 200 x 7 standardized PC scores
│   └── fig7(b)_product_feature_contributions.csv # 27 x 7 fused-PCA loading matrix
├── scripts_stata_prep/
│   ├── prepare_stata_data.py
│   └── fdr_correction.py
└── stata/
    ├── main_analysis.do
    └── main_analysis.log
```

## Data Availability and Privacy Boundary

The public dataset is product-level and anonymized. It does not contain raw
review text, user nicknames, user avatars, product URLs, raw product images,
shop names, or original brand names. The `brand` column in
`data_anonymized/product_master.*` is a two-letter anonymized code used only
for traceability within the research workflow.

`product_id` is formatted with an em dash, for example `1—1`, to avoid
spreadsheet software converting product identifiers into dates. If you run
custom scripts that expect a normal hyphen, normalize `—` to `-` before
merging.

## Expected Raw Review Excel Schema

The full pipeline expects a cleaned review workbook with the following schema.
The raw workbook itself is not released; researchers can prepare their own
workbook with the same column names and types to run the pipeline end-to-end.

| Column | Meaning | Required for |
|---|---|---|
| `excel文件名` | Source product file name beginning with the product code | product id fallback |
| `commentId` | Review identifier | deduplication, LLM extraction, aggregation |
| `no` | Product code, e.g. `1-1` | product id |
| `brand` | Brand name or anonymized brand code | metadata only |
| `price` | Product-page price | control variable |
| `favorable rate` | Platform favorable rate | satisfaction construction |
| `sales volume` | Product-page sales volume | optional robustness / control extension |
| `productId` | Platform product id | raw traceability; not used in public regression |
| `userNickName` | User nickname | not required for public analysis |
| `userImgURL` | User avatar URL | not required for public analysis |
| `buyCountText` | Purchase count text | not required for public analysis |
| `commentDate` | Review date / time | optional time-control extension |
| `commentData` | Cleaned review text | sentiment scoring and LLM extraction |

For minimum reproduction of the text pipeline, the necessary columns are
`no`, `commentId`, and `commentData`. To reconstruct satisfaction `Y`,
`price`, `favorable rate`, and `brand` are also required.

## Public Regression Reproduction (Path A)

The simplest reproduction path uses the shipped anonymized product-level data:

1. Open Stata (version 18 or compatible).
2. Set the working directory to the repository root.
3. Run:

```stata
do stata/main_analysis.do
```

The script reads `data_anonymized/product_master.dta` and writes output tables
to `stata/output/`. The released `stata/main_analysis.log` records a reference
run after local temporary paths were removed.

## Full Pipeline With User-Supplied Data (Path B)

If you supply your own cleaned reviews and product images, place them as
follows:

```text
data/input/product_reviews_clean.xlsx
data/images_1024/{product_id}.jpg
models/openaiclip-vit-large-patch14/
models/IDEA-CCNLTaiyi-CLIP-Roberta-large-326M-Chinese/
```

Then run the pipeline in this order:

```bash
python scripts_extraction/llm_sentiment_scoring.py
python scripts_satisfaction/compute_product_satisfaction.py
python scripts_extraction/llm_extract_kansei.py
python scripts_extraction/llm_extract_function.py
python scripts_extraction/compute_kansei_function_scores.py
python scripts_visual/encode_clip_images.py
python scripts_visual/encode_clip_text.py

# Apply the published fused-PCA loadings (see "Fused-PCA reduction" below)
# to produce data/processed/product_visual_pcs.csv with columns
# product_id, z_pc1, z_pc2, ..., z_pc7.

python scripts_stata_prep/prepare_stata_data.py
```

`prepare_stata_data.py` produces `data/processed/product_master.dta`. The
Stata script `main_analysis.do` automatically detects this file when the
shipped `data_anonymized/product_master.dta` is absent, so the two paths share
the same downstream analysis.

The LLM scripts use the DashScope OpenAI-compatible API endpoint and ask for
the API key at runtime. No API key is stored in code. Generated metadata
files may contain local paths and timestamps; do not commit generated
metadata, checkpoints, raw comments, or raw images if those contain
restricted information.

### Fused-PCA reduction

Between the visual encoding step (`encode_clip_text.py`) and the Stata
preparation step (`prepare_stata_data.py`), users must apply the published
fused-PCA loadings to a standardized concatenated visual matrix
(14 engineering one-hot + 8 image-PCA + 5 CLIP-text concept = 27 features per
product).

The figure-making PCA code is not released because the manuscript figures
were manually adjusted for layout, but the loading matrix is provided in two
machine-readable forms:

* `scripts_visual/fig7(b)_product_feature_contributions.csv` — long-format
  table of 27 features x 7 retained PCs, with per-feature standardized value,
  loading, and per-product contribution. The `loading` column reproduces the
  matrix used in the published regressions.
* The same matrix is documented in the paper's **Appendix C, Table C.1
  ("Full visual PCA loadings")**.

Apply these loadings to the standardized 27-dim fused vector, then z-score
each of the resulting 7 PCs to obtain `z_pc1`...`z_pc7`. Write the result to
`data/processed/product_visual_pcs.csv` for use by `prepare_stata_data.py`.

## Variable Construction Summary

### Affective and functional mediators

`llm_extract_kansei.py` extracts five affective evaluation dimensions
(`ME1`-`ME5`) from review text. `llm_extract_function.py` extracts seven
functional evaluation dimensions (`MF1`-`MF7`).
`compute_kansei_function_scores.py` converts comment-level matches into
product-level scores.

The scoring rules are:

```text
Affective: final_value = (-1 if negated else 1) * polarity * degree_weight
Function:  final_value = polarity * degree_weight
```

For functional dimensions, `polarity` is interpreted as the user's final
semantic attitude, so negation is retained as metadata but not mechanically
reversed. This handles cases such as "没有异味" being positive but
"没有反应" being negative.

### Satisfaction

`compute_product_satisfaction.py` constructs five product-level satisfaction
outcomes:

```text
Y_sentiment_only = sentiment_z
Y_favrate_only   = favrate_z
Y_50_50          = 0.5 * sentiment_z + 0.5 * favrate_z
Y_30_70          = 0.3 * sentiment_z + 0.7 * favrate_z
Y_70_30          = 0.7 * sentiment_z + 0.3 * favrate_z
```

`sentiment_z` is the z-score of product-level mean LLM sentiment. `favrate_z`
is the z-score of the logit-transformed platform favorable rate.

### Visual encoding

The visual side uses three layers:

1. `Engineering one-hot`: manually coded design attributes in
   `scripts_visual/produce_onehot.xlsx`.
2. `CLIP image embedding`: `encode_clip_images.py` extracts 768-dimensional
   image embeddings using `openai/clip-vit-large-patch14`.
3. `CLIP-text concept scoring`: `encode_clip_text.py` aligns product images
   with affective concept prompts using the Taiyi Chinese CLIP text encoder.

The final visual predictors used in regression are seven fused visual
principal components (`z_pc1`-`z_pc7`). Instead of the figure-making PCA
code, this repository ships the published PCA outputs needed for
reproduction:

```text
scripts_visual/fig7(a)_product_visual_pcs.csv             # 200 x 7 standardized PC scores
scripts_visual/fig7(b)_product_feature_contributions.csv  # 27 x 7 loading matrix
```

## Key Public Files

### `data_anonymized/product_master.csv`

This file contains the full product-level regression dataset:

- `product_id`
- `me1`-`me5`, `me1_z`-`me5_z`, `cov_me1`-`cov_me5`, `m_e_avg`, `m_e_avg_raw`
- `mf1`-`mf7`, `mf1_z`-`mf7_z`, `cov_mf1`-`cov_mf7`, `m_f_avg`, `m_f_avg_raw`
- `sentiment_mean`, `favorable_rate_raw`, `favorable_rate_logit`, `sentiment_z`, `favrate_z`
- `y_sentiment_only`, `y_favrate_only`, `y_50_50`, `y_30_70`, `y_70_30`
- `z_pc1`-`z_pc7`
- `brand` (anonymized code), `price`, `n_reviews_total`

### `scripts_visual/produce_onehot.xlsx`

This workbook contains the engineering one-hot coding sheet and coding rule
sheet. The public version retains product codes and design attributes but
does not include raw product URLs, original brand names, or identifiable
product images.

## Software Requirements

Install Python dependencies:

```bash
pip install -r requirements.txt
```

For CLIP image encoding, install a CUDA-enabled PyTorch build if GPU
acceleration is needed. Stata reproduction requires Stata 18 or a compatible
version with `estout / esttab` installed.

## Do Not Commit

The following files should not be committed to a public repository:

- raw review workbook with user text or platform identifiers
- raw or standardized product images
- generated API checkpoints and metadata if they contain local paths

## Method References

- Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). Learning transferable visual models from natural language supervision. *Proceedings of the 38th International Conference on Machine Learning*, 8748-8763. https://proceedings.mlr.press/v139/radford21a.html
- Gilardi, F., Alizadeh, M., & Kubli, M. (2023). ChatGPT outperforms crowd workers for text-annotation tasks. *Proceedings of the National Academy of Sciences*, 120(30), e2305016120. https://doi.org/10.1073/pnas.2305016120
- Abdi, H., & Williams, L. J. (2010). Principal component analysis. *WIREs Computational Statistics*, 2(4), 433-459. https://doi.org/10.1002/wics.101
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B (Methodological)*, 57(1), 289-300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x
- Hayes, A. F. (2017). *Introduction to mediation, moderation, and conditional process analysis: A regression-based approach* (2nd ed.). Guilford Press.
