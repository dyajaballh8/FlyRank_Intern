# Capstone Report — Search Performance Signals for Content Prioritization

- **Author:** Dyaa Abdullah
- **Lane:** CTR / Engagement Opportunity Scoring
- **Repo:** https://github.com/dyajaballh8/FlyRank_Intern
- **Date:** August 2026

## 0. Abstract

Content teams cannot manually review every page when search performance suggests a possible
opportunity, so they need a defensible way to decide what to look at first. This capstone asks
whether observable Google Search Console (GSC) signals — impressions, clicks, and average
position — can support that prioritization. Using the May 2026 release of
`fact_content_daily_performance` (4,237,020 rows across 55 pseudonymous clients), a transparent
threshold rule was compared against a Logistic Regression model on a client-grouped holdout
split. The rule baseline matched its own target definition perfectly (F1 = 1.0000), while
Logistic Regression reached F1 = 0.6584 with high recall (0.9908) but modest precision (0.4930).
The output is a ranked, reason-coded review queue intended strictly for human decision support —
not an automatic content-editing system, and not proof that any specific change improves CTR.

## 1. Problem framing

**Decision supported:** which content items a human reviewer should look at first when search
data suggests a possible CTR issue.

**Unit of analysis:** one content item on one report date (a `content_hash_id` × `report_date`
row in the daily performance table).

**Output:** a ranked action queue, where each row carries a priority level (High / Medium / Low)
and a reason code explaining why it was surfaced.

**Action a human takes:** review the title, meta description, and search snippet for
priority items, and confirm the recommendation is supported by context (search intent, brand vs.
non-brand traffic, recent edits) before making any change.

**Cost of a wrong call:** a false positive wastes reviewer time on a page that didn't need it; a
false negative lets a real opportunity go unreviewed. Neither is catastrophic on its own, which
is why the system is framed as prioritization support rather than an automated gate.

**Why ML helps here:** a single fixed threshold treats every borderline case the same way. A
model that weighs impressions, clicks, and position together can rank borderline items more
usefully than a rule can, even though (as the leakage audit below shows) that ranking still needs
to be read carefully.

## 2. Data safety

**Source:** FlyRank ML Internship warehouse (Hugging Face, gated access), table
`fact_content_daily_performance`, May 2026 release (`month=2026-05/data_0.parquet`).

**Columns used:** `client_hash_id` (grouping only, never a feature), `content_hash_id`,
`report_date`, `gsc_impressions`, `gsc_clicks`, `gsc_avg_position`.

**Filters applied:** rows kept only where `gsc_data_available = TRUE`, `gsc_impressions > 0`, and
`gsc_avg_position > 0`. Rows without usable GSC measurements were excluded because both the rule
and the model depend entirely on these three signals.

**What was deliberately excluded:** no client names, domains, URLs, private queries,
credentials, or raw exports appear anywhere in `work/`. `client_hash_id` is used only to build
the grouped train/test split — it is never passed to the model as a feature.

**Leakage risk considered and disclosed:** the target label is built from the same three signals
(`gsc_impressions`, `gsc_avg_position`, `gsc_clicks` via CTR) that are also the model's only
features. This is a real circularity, and it is called out explicitly in Section 5 and in the
validation notebook rather than hidden behind a strong-looking score.

## 3. Baseline

The Week-4 rule flags a content item as a CTR opportunity when:

- `gsc_impressions >= 500`
- `1 <= gsc_avg_position <= 20`
- `CTR < 0.5%` (CTR = clicks / impressions × 100)

This is a fair comparison point because it uses the exact same three observable signals as the
model, with no hidden information advantage on either side. Evaluated on the grouped holdout, the
rule's predictions matched the target definition exactly:

| Metric | Value |
|---|---:|
| Accuracy | 1.0000 |
| Precision | 1.0000 |
| Recall | 1.0000 |
| F1 | 1.0000 |

This perfect score is expected, not impressive — the rule *is* the target definition, so it
cannot miss. It is reported here as the honest reference point, not as evidence the problem is
solved (see Section 5).

## 4. Model / analysis

**Method:** Logistic Regression (scikit-learn), with `StandardScaler` preprocessing and
`class_weight="balanced"` to account for the rare positive class.

**Features (exactly three, all observable at decision time):**
- `gsc_impressions`
- `gsc_clicks`
- `gsc_avg_position`

**Deliberately left out:** any product/business decision flags, any future-window signal, and
any content or client identifier. Nothing beyond the three GSC signals above was made available
to the model.

**Target definition (one sentence):** a row is labeled `1` ("CTR review opportunity") when
impressions ≥ 500, average position is between 1 and 20, and CTR is below 0.5%; otherwise `0`.

**Base rate:** the positive class is rare — 67,662 of 4,237,020 rows (about 1.6%) — which is why
F1 rather than accuracy is the headline metric throughout.

## 5. Evaluation

**Split:** grouped by `client_hash_id` using `GroupShuffleSplit` (80/20, `random_state=42`).
Content from the same client can share similar traffic patterns, so a random row-level split
would let the same client appear in both train and test, making the evaluation optimistic. The
grouped split guarantees zero client overlap: 44 clients / 3,336,055 rows in train, 11 clients /
900,965 rows in test.

**Model vs. baseline, same split, same metric:**

| Method | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Week-4 Rule Baseline | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Logistic Regression | 0.9867 | 0.4930 | 0.9908 | 0.6584 |

**Naive vs. grouped split (honesty check):** evaluating the same Logistic Regression model on a
naive row-level split gave an inflated F1 of 0.7224, versus 0.6584 on the client-grouped split —
a 0.064 drop once client leakage between train and test is removed. The grouped number is the one
reported as the model's real performance.

**Error analysis:** on the test split, the model produced 11,914 false positives and 108 false
negatives out of 900,965 rows. The false-negative rate is low (the model rarely misses a true
opportunity — consistent with its 0.99 recall), but false positives are common, consistent with
its 0.49 precision: many flagged items turn out not to meet the target definition once thresholds
are applied strictly. A reviewer using this queue should expect roughly half of "flagged" items to
be false alarms, which is why every recommendation is framed as "worth a look," not "confirmed."

**Leakage audit:** all three model features are also the exact signals used to construct the
target (`gsc_impressions`, `gsc_avg_position`, `gsc_clicks`). This circular dependency is the
single biggest caveat on these numbers — see Section 6.

## 6. Interpretation

Logistic Regression traded the rule's perfect-but-circular precision for much higher recall at
lower precision — it catches nearly every item the rule would flag (0.9908 recall) but also
raises many extra candidates (0.4930 precision). In plain words: the model over-flags. That is
a reasonable trade for a *prioritization* tool (better to over-surface than to silently miss a
real opportunity), but it is a poor trade if the queue were ever used to trigger anything
automatic.

The most important negative result is the leakage finding itself: because the target is built
directly from the same three signals the model consumes, no score here should be read as
evidence that the model understands *why* a page underperforms on CTR — only that it can
reproduce a rule-like pattern in the same signals it was given. A stronger design for a future
iteration would define the target from a separate future outcome window (e.g., "did CTR improve
in the next reporting period") and keep today's signals as the only inputs — genuinely separating
what the model sees from what it is scored against.

## 7. Recommendation

The action playbook converts the model's scores into a ranked, reason-coded queue for human
review (4,237,020 rows total, all pseudonymous):

| Priority | Reason code | Rows | Recommended action |
|---|---|---:|---|
| High | `HIGH_IMPRESSIONS_LOW_CTR` | 67,662 | Review title, meta description, and search snippet — the strongest CTR-review candidates |
| Medium | `GOOD_POSITION_LOW_CTR` | 2,296,420 | Review search intent, title, and snippet alignment |
| Medium | `LOW_VISIBILITY` | 1,566,164 | Broader review of content relevance and visibility |
| Low | `GENERAL_REVIEW` | 306,774 | Lower-priority monitoring candidate |

**How a FlyRank editor would use this tomorrow:** start with the High-priority queue, confirm
each item's search intent and current snippet before touching anything, and treat everything
below High as a backlog rather than an urgent list.

**Confidence and limits, stated plainly:** this is decision support, not an automation system.
No item should be edited, redirected, merged, or deleted based on this queue alone — human
review against real editorial and business context is required first (see the no-go list in
`work/notebooks/w07_action_playbook.ipynb`, Section 3). Given the precision of 0.49, expect
roughly half of "flagged" items to not actually warrant a change once reviewed.

## 8. Reproducibility

**Fresh clone → same result:**

```bash
git clone https://github.com/dyajaballh8/FlyRank_Intern
cd FlyRank_Intern
pip install -r requirements.txt
```

Open `work/notebooks/w05_model.ipynb` (model vs. baseline), `work/notebooks/w06_validation_audit.ipynb`
(naive-vs-grouped split and leakage audit), `work/notebooks/w07_action_playbook.ipynb` (ranked
queue export), and `work/notebooks/capstone.ipynb` (this report's source notebook) in Google
Colab via their badges, or locally in Jupyter. Each notebook re-authenticates to the Hugging Face
warehouse with a personal `HF_TOKEN` and re-reads the same `month=2026-05` release, so the
numbers above regenerate from the same source data.

**Random seed:** `random_state=42` throughout (`GroupShuffleSplit`, `LogisticRegression`).

**Environment:** `pandas`, `duckdb`, `scikit-learn`, `numpy`, `matplotlib` — see
`requirements.txt` at the repo root.

**Generated artifacts:** `work/notebooks/w07_action_playbook.ipynb` writes
`work/outputs/w07_ranked_action_queue.csv` and `work/outputs/w07_action_playbook_metrics.json`;
`work/notebooks/capstone.ipynb` writes `work/figures/model_vs_baseline_f1.png`. These are
generated data artifacts and are intentionally kept out of Git per the repo's rules — re-run the
notebooks to regenerate them.

## 9. Acknowledgments & data credit

Built on the FlyRank ML Internship dataset — [flyrank.ai](https://flyrank.ai).

This work was completed as part of the FlyRank Machine Learning Internship capstone. All
reported figures are aggregated and pseudonymized; no client names, domains, private queries,
credentials, or other private identifiers appear anywhere in this report or in `work/`.

---

> **Claims checklist:** observed / measured / directional / decision-support language used
> throughout · base rate reported (1.6% positive class) alongside precision/recall · no causal
> claims · no claim to predict or reverse-engineer Google's ranking algorithm · no
> client-identifying details anywhere · every number above matches the notebooks in
> `work/notebooks/` as committed to this repo.
