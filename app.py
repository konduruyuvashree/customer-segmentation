import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation AI",
    page_icon="🧠",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0a0a0f;
    color: #e8e4d9;
}
h1, h2, h3 { font-family: 'Syne', sans-serif; font-weight: 800; }
.stButton>button {
    background: linear-gradient(135deg, #c9a84c, #f0d080);
    color: #0a0a0f;
    border: none;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    padding: 0.6rem 2rem;
    border-radius: 2px;
    letter-spacing: 0.05em;
}
.stButton>button:hover { opacity: 0.85; }
.metric-card {
    background: #13131a;
    border: 1px solid #2a2a3a;
    border-left: 4px solid #c9a84c;
    padding: 1.2rem 1.5rem;
    border-radius: 4px;
    margin-bottom: 1rem;
}
.insight-card {
    background: #13131a;
    border: 1px solid #2a2a3a;
    padding: 1.2rem 1.5rem;
    border-radius: 4px;
    margin-bottom: 0.8rem;
}
.cluster-tag {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 2px;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    margin-right: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown("## 🧠 Customer Segmentation AI")
st.markdown("##### K-Means Clustering · Business Intelligence Dashboard")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    data_source = st.radio("Data Source", ["Use Sample Dataset", "Upload CSV"])
    k = st.slider("Number of Clusters (K)", min_value=2, max_value=8, value=4)
    features = st.multiselect(
        "Features to Cluster On",
        ["Age", "Annual Income (k$)", "Spending Score (1-100)"],
        default=["Annual Income (k$)", "Spending Score (1-100)"]
    )
    st.markdown("---")
    st.markdown("**ℹ️ About**")
    st.caption("Groups customers by behavior using K-Means. Each cluster gets actionable business insights.")

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_sample_data():
    np.random.seed(42)
    n = 200
    data = {
        "CustomerID": range(1, n+1),
        "Gender": np.random.choice(["Male", "Female"], n),
        "Age": np.random.randint(18, 70, n),
        "Annual Income (k$)": np.random.randint(15, 140, n),
        "Spending Score (1-100)": np.random.randint(1, 100, n),
    }
    return pd.DataFrame(data)

df = None
if data_source == "Use Sample Dataset":
    df = load_sample_data()
    st.success("✅ Sample Mall Customer Dataset loaded (200 customers)")
else:
    uploaded = st.file_uploader("Upload your CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"✅ Uploaded: {uploaded.name} — {len(df)} rows")

if df is None:
    st.info("👈 Select a data source from the sidebar to begin.")
    st.stop()

# ── Preview ───────────────────────────────────────────────────────────────────
with st.expander("📋 Preview Dataset"):
    st.dataframe(df.head(10), use_container_width=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", len(df))
    col2.metric("Features", len(df.columns))
    col3.metric("Clusters Selected", k)

st.markdown("---")

# ── Run Clustering ────────────────────────────────────────────────────────────
if not features:
    st.warning("Please select at least one feature from the sidebar.")
    st.stop()

missing = [f for f in features if f not in df.columns]
if missing:
    st.error(f"These features not found in dataset: {missing}")
    st.stop()

X = df[features].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method
inertias = []
K_range = range(2, 10)
for ki in K_range:
    km = KMeans(n_clusters=ki, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

# Final KMeans
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df_clean = df.loc[X.index].copy()
df_clean["Cluster"] = kmeans.fit_predict(X_scaled)

# ── Cluster Colors ────────────────────────────────────────────────────────────
COLORS = ["#c9a84c", "#e05252", "#52b0e0", "#7ed87e", "#b07edc", "#e09c52", "#e0d452", "#52e0c4"]
cluster_colors = {i: COLORS[i % len(COLORS)] for i in range(k)}

# ── Business Insights ─────────────────────────────────────────────────────────
def generate_insight(cluster_id, stats):
    income_col = "Annual Income (k$)"
    score_col = "Spending Score (1-100)"
    age_col = "Age"

    has_income = income_col in stats.index
    has_score = score_col in stats.index
    has_age = age_col in stats.index

    income = stats.get(income_col, 50)
    score = stats.get(score_col, 50)
    age = stats.get(age_col, 35)

    if has_income and has_score:
        if income > 70 and score > 60:
            label = "💎 Premium Loyalists"
            strategy = "Target with exclusive offers, loyalty rewards, and premium product launches."
        elif income > 70 and score <= 60:
            label = "💼 Cautious High Earners"
            strategy = "Send value-proof content, ROI-focused campaigns, and trust-building offers."
        elif income <= 70 and score > 60:
            label = "🛍️ Enthusiastic Bargain Hunters"
            strategy = "Flash sales, discount codes, and FOMO-driven promotions work best."
        else:
            label = "😴 Low-Engagement Segment"
            strategy = "Re-engage with personalized outreach, surveys, and win-back campaigns."
    else:
        label = f"Cluster {cluster_id}"
        strategy = "Analyze further for targeted strategies."

    return label, strategy

# ── Cluster Summary ───────────────────────────────────────────────────────────
st.markdown("### 📊 Cluster Summary & Business Insights")

cluster_stats = df_clean.groupby("Cluster")[features].mean()

cols = st.columns(min(k, 4))
for i in range(k):
    with cols[i % len(cols)]:
        stats = cluster_stats.loc[i]
        label, strategy = generate_insight(i, stats)
        count = len(df_clean[df_clean["Cluster"] == i])

        income_val = "${:.0f}k".format(stats["Annual Income (k$)"]) if "Annual Income (k$)" in features else "N/A"
        score_val = "{:.0f}".format(stats["Spending Score (1-100)"]) if "Spending Score (1-100)" in features else "N/A"
        age_val = "{:.0f}".format(stats["Age"]) if "Age" in features else "N/A"

        st.markdown(f"**C{i} — {label}**")
        st.markdown(f"👥 **{count}** customers")
        if "Annual Income (k$)" in features:
            st.markdown(f"💰 Income: **{income_val}**")
        if "Spending Score (1-100)" in features:
            st.markdown(f"⭐ Score: **{score_val}**")
        if "Age" in features:
            st.markdown(f"🎂 Age: **{age_val}**")
        st.info(f"📌 {strategy}")
        st.markdown("")

# ── Plots ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📈 Visualizations")

plot_col1, plot_col2 = st.columns(2)

# Scatter Plot
with plot_col1:
    st.markdown("**Cluster Distribution**")
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0a0a0f")
    ax.set_facecolor("#13131a")
    if len(features) >= 2:
        for i in range(k):
            mask = df_clean["Cluster"] == i
            ax.scatter(
                df_clean.loc[mask, features[0]],
                df_clean.loc[mask, features[1]],
                c=cluster_colors[i], s=40, alpha=0.8, label=f"C{i}"
            )
        ax.set_xlabel(features[0], color="#aaa")
        ax.set_ylabel(features[1], color="#aaa")
    else:
        ax.hist(df_clean[features[0]], bins=20, color=COLORS[0])
        ax.set_xlabel(features[0], color="#aaa")

    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2a3a")
    ax.legend(facecolor="#0a0a0f", edgecolor="#2a2a3a", labelcolor="#e8e4d9")
    st.pyplot(fig)

# Elbow Curve
with plot_col2:
    st.markdown("**Elbow Method (Optimal K)**")
    fig2, ax2 = plt.subplots(figsize=(6, 4), facecolor="#0a0a0f")
    ax2.set_facecolor("#13131a")
    ax2.plot(list(K_range), inertias, color="#c9a84c", marker="o", linewidth=2, markersize=6)
    ax2.axvline(x=k, color="#e05252", linestyle="--", linewidth=1.5, label=f"Selected K={k}")
    ax2.set_xlabel("Number of Clusters (K)", color="#aaa")
    ax2.set_ylabel("Inertia", color="#aaa")
    ax2.tick_params(colors="#aaa")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#2a2a3a")
    ax2.legend(facecolor="#0a0a0f", edgecolor="#2a2a3a", labelcolor="#e8e4d9")
    st.pyplot(fig2)

# Cluster Size Bar
st.markdown("**Cluster Size Distribution**")
fig3, ax3 = plt.subplots(figsize=(10, 3), facecolor="#0a0a0f")
ax3.set_facecolor("#13131a")
sizes = df_clean["Cluster"].value_counts().sort_index()
bars = ax3.bar(
    [f"Cluster {i}" for i in sizes.index],
    sizes.values,
    color=[cluster_colors[i] for i in sizes.index],
    edgecolor="#0a0a0f", linewidth=1.5
)
for bar, val in zip(bars, sizes.values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             str(val), ha='center', color="#e8e4d9", fontsize=10)
ax3.tick_params(colors="#aaa")
for spine in ax3.spines.values():
    spine.set_edgecolor("#2a2a3a")
st.pyplot(fig3)

# ── Download ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 💾 Export Results")
csv = df_clean.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Segmented Data (CSV)",
    data=csv,
    file_name="customer_segments.csv",
    mime="text/csv"
)

st.markdown("---")
st.caption("Built with K-Means · Scikit-learn · Streamlit | BTech AI/ML Project")
