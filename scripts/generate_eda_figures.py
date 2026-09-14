"""
Generates publication-grade, high-resolution visual assets for Phase 1 EDA & Reports.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

WORKSPACE_DIR = r"x:\Projects\srm thevidyas event"
FIGURES_DIR = os.path.join(WORKSPACE_DIR, "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Set high-aesthetic style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['figure.dpi'] = 300

# Load cleaned data
posts_path = os.path.join(WORKSPACE_DIR, "Social_Engine_Posts_Cleaned.csv")
users_path = os.path.join(WORKSPACE_DIR, "Social_Engine_Users_Normalized.csv")
df_posts = pd.read_csv(posts_path)
df_users = pd.read_csv(users_path)
df_merged = df_posts.merge(df_users, on='user_id', how='left')

# Palette
ACCENT_PALETTE = ["#2563EB", "#7C3AED", "#EC4899", "#10B981", "#F59E0B", "#6366F1", "#14B8A6"]

# Figure 1: Intake Pipeline Corruption Breakdown (Waterfall / Bar Chart)
fig, ax = plt.subplots(figsize=(10, 5.5))
corruption_categories = [
    'Raw Input Records',
    'Exact Duplicates Removed',
    'Sign-Inverted Likes Fixed',
    'Missing Likes Imputed',
    'Pseudo-Null Texts Fixed',
    'Missing Platforms Imputed',
    'Cleaned Target Records'
]
counts = [12360, 360, 509, 1814, 72, 1784, 12000]
colors = ['#64748B', '#EF4444', '#F59E0B', '#F59E0B', '#3B82F6', '#8B5CF6', '#10B981']

bars = ax.barh(corruption_categories[::-1], counts[::-1], color=colors[::-1], edgecolor='none', height=0.6)
ax.set_title("Intake Pipeline Restoration: Corruption & Repair Breakdown", pad=15)
ax.set_xlabel("Number of Post Records")
for bar in bars:
    w = bar.get_width()
    ax.text(w + 150, bar.get_y() + bar.get_height()/2, f"{int(w):,}", ha='left', va='center', fontweight='bold', color='#1E293B')
ax.set_xlim(0, 14000)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "01_intake_corruption_breakdown.png"))
plt.close(fig)

# Figure 2: Engagement Distribution (Likes, Shares, Comments) across Platforms
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
platforms = df_posts['platform'].value_counts().index.tolist()

sns.boxplot(data=df_posts, x='platform', y='likes', ax=axes[0], palette="Blues_d", showmeans=True,
            meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"})
axes[0].set_title("Likes Distribution by Platform")
axes[0].set_ylabel("Likes")
axes[0].tick_params(axis='x', rotation=30)

sns.boxplot(data=df_posts, x='platform', y='shares', ax=axes[1], palette="Purples_d", showmeans=True,
            meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"})
axes[1].set_title("Shares Distribution by Platform")
axes[1].set_ylabel("Shares")
axes[1].tick_params(axis='x', rotation=30)

sns.boxplot(data=df_posts, x='platform', y='comments', ax=axes[2], palette="Greens_d", showmeans=True,
            meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"})
axes[2].set_title("Comments Distribution by Platform")
axes[2].set_ylabel("Comments")
axes[2].tick_params(axis='x', rotation=30)

plt.suptitle("Comparative Engagement Metrics by Platform (N = 12,000)", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "02_engagement_distribution_by_platform.png"), bbox_inches='tight')
plt.close(fig)

# Figure 3: Brand Perception & Sentiment Analysis
brand_sentiment = pd.crosstab(df_posts['brand'], df_posts['sentiment'], normalize='index') * 100
# Filter out generic/other for clean brand view
brand_sentiment = brand_sentiment.loc[[b for b in brand_sentiment.index if 'Generic' not in b]]
order_sent = ['Positive', 'Neutral', 'Negative']
brand_sentiment = brand_sentiment[order_sent]

fig, ax = plt.subplots(figsize=(12, 6))
brand_sentiment.plot(kind='barh', stacked=True, color=['#10B981', '#94A3B8', '#EF4444'], ax=ax, edgecolor='white', width=0.7)
ax.set_title("Brand Sentiment Share: Customer Feedback Profile (% Distribution)", pad=15)
ax.set_xlabel("Sentiment Percentage (%)")
ax.set_ylabel("Brand")
ax.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc='upper left')
for c in ax.containers:
    ax.bar_label(c, fmt='%.1f%%', label_type='center', color='white', fontweight='bold', fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "03_brand_sentiment_profile.png"), bbox_inches='tight')
plt.close(fig)

# Figure 4: Temporal Posting Dynamics & Day-of-Week Hourly Heatmap
df_posts['timestamp_dt'] = pd.to_datetime(df_posts['timestamp'])
df_posts['hour'] = df_posts['timestamp_dt'].dt.hour
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
heatmap_data = df_posts.pivot_table(index='day_of_week', columns='hour', values='post_id', aggfunc='count').reindex(day_order)

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='d', cbar_kws={'label': 'Post Volume'}, ax=ax, linewidths=.5)
ax.set_title("Temporal Engine Activity: Post Volume by Day and Hour of Day", pad=15)
ax.set_xlabel("Hour of Day (UTC)")
ax.set_ylabel("Day of Week")
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "04_hourly_dayofweek_activity_heatmap.png"), bbox_inches='tight')
plt.close(fig)

# Figure 5: Top 15 Cities by User Engagement Rate
city_engagement = df_merged.groupby('location').agg(
    total_posts=('post_id', 'count'),
    total_engagement=('total_engagement', 'sum'),
    avg_engagement_rate=('engagement_rate_per_follower', 'mean'),
    avg_followers=('follower_count', 'mean')
).sort_values('avg_engagement_rate', ascending=False).head(15)

fig, ax1 = plt.subplots(figsize=(12, 6.5))
y_pos = np.arange(len(city_engagement))
ax1.barh(y_pos, city_engagement['avg_engagement_rate'], color='#2563EB', alpha=0.85, height=0.6)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(city_engagement.index)
ax1.invert_yaxis()
ax1.set_xlabel("Average Engagement Rate per Follower", color='#2563EB')
ax1.set_title("Global Hubs: Top 15 Geographic Locations by Follower Engagement Rate", pad=15)
for i, v in enumerate(city_engagement['avg_engagement_rate']):
    ax1.text(v + 0.005, i, f"{v:.3f}", va='center', color='#1E293B', fontweight='bold', fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "05_top_cities_engagement_rate.png"), bbox_inches='tight')
plt.close(fig)

# Figure 6: Engagement Correlation Matrix
fig, ax = plt.subplots(figsize=(8, 6.5))
corr_cols = ['likes', 'shares', 'comments', 'total_engagement', 'virality_score', 'follower_count']
corr = df_merged[corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".3f", cmap="vlag", vmin=-0.1, vmax=1.0, mask=mask, square=True, linewidths=1, ax=ax)
ax.set_title("Correlation Matrix of Normalized Metrics", pad=15)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "06_correlation_matrix.png"), bbox_inches='tight')
plt.close(fig)

print("All 6 publication-quality figures successfully generated in reports/figures/!")
