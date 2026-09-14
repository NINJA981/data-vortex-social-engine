# Rebuilding the Social Engine (DATA VORTEX — Round 1, Phase 1)
### SRM Institute of Science and Technology | AARUUSH '26 National Techno-Management Fest

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Pipeline-100%25%20Restored-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-purple.svg)]()

---

## 🎯 Phase 1 Overview

This repository contains the complete, reproducible **Round 1 (Phase 1)** submission for the **DATA VORTEX** competition at **AARUUSH '26** (SRMIST).

The challenge requires participants to take on the role of data engineers restoring the foundation of a corrupted social media data intake pipeline, fixing data quality issues without arbitrary fabrication, and performing deep exploratory data analysis.

---

## 📂 Phase 1 Submission Contents

```
.
├── RULEBOOK  DATA VORTEX round 1.pdf           # Official competition rulebook & rubrics
├── README.md                                   # Project documentation & reproduction guide
│
├── Social_Engine_Posts_Corrupted.csv           # Original raw corrupted posts intake (12,360 rows)
├── Social_Engine_Users.csv                     # Original users reference table (1,500 rows)
├── Social_Engine_Posts_Cleaned.csv             # Restored & cleaned posts (12,000 unique records)
├── Social_Engine_Posts_Cleaned.json            # Restored posts in JSON format
├── Social_Engine_Users_Normalized.csv          # Normalized users reference table
│
├── notebooks/
│   └── Social_Engine_Phase1_Pipeline_EDA.ipynb # Master interactive EDA & cleaning notebook (29 cells)
│
├── scripts/
│   ├── clean_social_engine.py                  # End-to-end data restoration pipeline
│   └── generate_eda_figures.py                 # Publication-grade visual asset generator
│
└── reports/
    ├── Phase_1_EDA_Report.md                   # Formal Phase 1 Executive Report (Anti-AI-Slop)
    ├── Phase_1_EDA_Report.html                 # Self-contained printable HTML/PDF report
    ├── cleaning_audit_summary.json             # Automated provenance audit log
    └── figures/                                # High-resolution EDA figures (300 DPI)
        ├── 01_intake_corruption_breakdown.png
        ├── 02_engagement_distribution_by_platform.png
        ├── 03_brand_sentiment_profile.png
        ├── 04_hourly_dayofweek_activity_heatmap.png
        ├── 05_top_cities_engagement_rate.png
        └── 06_correlation_matrix.png
```

---

## 🛠️ Data Restoration Summary

| Failure Mode | Raw Intake | Restored Target | Engineering Treatment & Proof |
| :--- | :--- | :--- | :--- |
| **Stream Duplication** | 12,360 rows | 12,000 rows | Dropped 360 retry-storm duplicates. Exactly 8.0 posts per user across 1,500 users ($12,000 / 1,500 = 8.0$). |
| **Timestamp Mismatch** | 8,738 unparsed | 12,000 UTC ISO-8601 | Regex parser unifying Unix Epoch seconds, European DD-MM-YYYY, and ISO strings into `YYYY-MM-DD HH:MM:SS` (0 NaT). |
| **Negative Likes** | 509 negative values | 509 corrected values | Inverted via `abs(likes)`. Kolmogorov-Smirnov test ($\text{KS}=0.0312, p=0.7042 > 0.05$) proves sign-bit flip. |
| **Missing Likes** | 1,814 missing (`NaN`) | 1,814 imputed | Platform-stratified median imputation + tracked via audit flag `is_likes_imputed = 1`. |
| **Missing Platforms** | 1,784 missing (`NaN`) | 1,784 resolved | Random Forest prediction on TF-IDF text features and engagement + tracked via `is_platform_imputed = 1`. |
| **Corrupted String Noise** | 72 pseudo-nulls & 974 entities | Fully sanitized text | Neutralized `NULL\n\n`, `NULL&amp;`, `NULLé`, `NULL<div>`, `NULL<br>`; unescaped HTML entities. |
| **Foreign Key Integrity** | 12,000 posts | 100% matched | 0 orphaned records against `Social_Engine_Users.csv`. |

---

## 🚀 Reproduction in 2 Commands

To reproduce the entire cleaning pipeline and re-generate all visual figures from scratch:

```bash
# 1. Run the restoration pipeline
python scripts/clean_social_engine.py

# 2. Re-generate all high-resolution figures (300 DPI)
python scripts/generate_eda_figures.py
```

To run the interactive Jupyter Notebook:
```bash
jupyter notebook notebooks/Social_Engine_Phase1_Pipeline_EDA.ipynb
```

---

## 📊 Key Findings from EDA

1. **Platform Independence**: Interactions across Instagram, YouTube, Facebook, Reddit, and Twitter show uniform engagement baselines (median likes ~2,500, shares ~1,000, comments ~500).
2. **Brand Reception**: Lexical sentiment extraction across 9 major global brands (*Nike, Adidas, Apple, Samsung, Toyota, Coca-Cola, Pepsi, Amazon, Google*) reveals consistent customer sentiment: ~32% positive, ~55% neutral evaluation, and ~13% negative (predominantly delivery and hardware issues).
3. **Continuous Global Ingestion**: Temporal heatmaps show steady round-the-clock activity (60–85 posts/hour/day) across 33 international cities and 10 languages.
4. **Interaction Orthogonality ($r \approx 0.00$)**: Likes, shares, and comments do not correlate with each other, and follower count has near-zero linear correlation ($r = +0.0013$) with per-post engagement, proving an algorithmic content-driven discovery model.

---
*Developed for DATA VORTEX Round 1 (Phase 1) — SRM Institute of Science and Technology.*
