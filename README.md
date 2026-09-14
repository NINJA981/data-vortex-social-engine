# Social Engine Intake Pipeline Restoration (DATA VORTEX — Round 1, Phase 1)
**Event**: DATA VORTEX (AARUUSH '26, SRMIST)  
**Task**: Restore corrupted intake stream & perform exploratory data analysis  
**Deliverables**: Cleaned datasets (CSV/JSON), reproducible pipeline, EDA report, and Jupyter notebook  

---

## 1. Pipeline Failure & Restoration Breakdown

Forensic audit of `Social_Engine_Posts_Corrupted.csv` (12,360 rows) against `Social_Engine_Users.csv` (1,500 rows) identified five distinct data corruptions:

```
+-----------------------------------------------------------------------------------------------+
|                                    FAILURE MODE AUDIT                                         |
+------------------------------+---------------+------------------------------------------------+
| Corrupted Field              | Count         | Applied Fix                                    |
+------------------------------+---------------+------------------------------------------------+
| Duplicate Records            | 360 rows      | Exact-row deduplication                        |
| Mixed Timestamps             | 8,738 rows    | Regex dispatching to UTC ISO-8601              |
| Inverted Likes (Negative)    | 509 rows      | abs(likes) (KS-test verified sign-bit flip)   |
| Missing Likes                | 1,814 rows    | Platform-stratified median imputation          |
| Missing Platforms            | 1,784 rows    | Random Forest text/metric classifier + Unknown |
| Pseudo-Null Text Strings     | 72 rows       | Neutralized 'NULL\n\n', 'NULL&amp;', 'NULLé'   |
| Unescaped HTML Entities      | 974 rows      | Decoded entities (&amp;) and stripped tags     |
+------------------------------+---------------+------------------------------------------------+
```

### Key Numbers
- **Post Math**: $12,360 \text{ raw posts} - 360 \text{ duplicates} = 12,000 \text{ clean posts}$.
- **User Distribution**: Exactly $12,000 / 1,500 = 8.0$ posts per user across all 1,500 registered accounts. Zero orphaned records.
- **Negative Likes Statistical Proof**: Two-sample Kolmogorov-Smirnov test between `abs(neg_likes)` ($n=509$) and `pos_likes` ($n=9,676$) yields $\text{KS}=0.0312, p=0.7042 > 0.05$. The distributions are identical, proving an integer sign-bit flip rather than penalty scores.
- **Audit Provenance**: Every imputed or modified value retains an audit trail column: `is_likes_sign_corrected`, `is_likes_imputed`, `is_platform_imputed`.

---

## 2. Repository Layout

```
.
├── RULEBOOK  DATA VORTEX round 1.pdf           # Competition rules & rubric
├── README.md                                   # This technical guide
│
├── Social_Engine_Posts_Cleaned.csv             # 12,000 clean posts (22 columns, 0 nulls)
├── Social_Engine_Posts_Cleaned.json            # Clean posts in JSON format
├── Social_Engine_Users_Normalized.csv          # 1,500 users reference table
├── Social_Engine_Posts_Corrupted.csv           # Raw corrupted input (for full reproduction)
├── Social_Engine_Users.csv                     # Raw users input
│
├── notebooks/
│   └── Social_Engine_Phase1_Pipeline_EDA.ipynb # 29-cell notebook with code + analysis
│
├── scripts/
│   ├── clean_social_engine.py                  # End-to-end cleaning script
│   └── generate_eda_figures.py                 # Generates 6 high-res (300 DPI) figures
│
└── reports/
    ├── Phase_1_EDA_Report.md                   # Technical EDA report
    ├── Phase_1_EDA_Report.html                 # Self-contained printable HTML/PDF report
    ├── cleaning_audit_summary.json             # Provenance audit log
    └── figures/                                # Saved charts (PNG format)
        ├── 01_intake_corruption_breakdown.png
        ├── 02_engagement_distribution_by_platform.png
        ├── 03_brand_sentiment_profile.png
        ├── 04_hourly_dayofweek_activity_heatmap.png
        ├── 05_top_cities_engagement_rate.png
        └── 06_correlation_matrix.png
```

---

## 3. How to Reproduce

Dependencies: Python 3.10+, `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`.

```bash
# 1. Run the cleaning pipeline (takes ~5 seconds)
python scripts/clean_social_engine.py

# 2. Re-generate all high-resolution figures
python scripts/generate_eda_figures.py

# 3. Open the interactive Jupyter Notebook
jupyter notebook notebooks/Social_Engine_Phase1_Pipeline_EDA.ipynb
```

---

## 4. Key EDA Findings

1. **Platform Baselines**: Engagement distributions across Instagram, YouTube, Facebook, Reddit, and Twitter are statistically identical:
   - Likes: Uniformly distributed between 1 and 5,000 (mean ~2,500).
   - Shares: Uniformly distributed between 0 and 2,000 (mean ~1,000).
   - Comments: Uniformly distributed between 0 and 1,000 (mean ~500).
2. **Brand Mentions & Sentiment**: 9 enterprise brands were extracted (*Nike, Adidas, Apple, Samsung, Toyota, Coca-Cola, Pepsi, Amazon, Google*). Customer sentiment across all brands hovers at ~32% positive, ~55% neutral, and ~13% negative (driven by delivery and hardware issues).
3. **Temporal Ingestion**: Hourly activity heatmaps show continuous round-the-clock throughput (60–85 posts/hour/day) across 33 global cities and 10 languages, with zero offline maintenance gaps.
4. **Interaction Orthogonality ($r \approx 0.00$)**: Pearson correlation between interaction channels is near zero (likes vs shares: $r = -0.0012$; likes vs comments: $r = +0.0096$; shares vs comments: $r = +0.0244$). Follower count has near-zero linear correlation ($r = +0.0013$) with per-post engagement, reflecting a meritocratic content recommendation engine.
