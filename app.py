import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Customer Segmentation AI", page_icon="🧠", layout="wide")

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
    st.markdown("**Navigation**")
    section = st.radio("Go to", [
        "Cluster Summary",
        "Visualizations",
        "ML Explanation",
        "Predict New Customer",
        "Algorithm Comparison",
        "Export"
    ])
    st.markdown("---")
    st.caption("K-Means · PCA · Silhouette · BTech AI/ML Project")

st.markdown("## 🧠 Customer Segmentation AI")
st.markdown("##### K-Means Clustering · Business Intelligence Dashboard")
st.markdown("---")

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

with st.expander("📋 Preview Dataset"):
    st.dataframe(df.head(10), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Customers", len(df))
    c2.metric("Features", len(df.columns))
    c3.metric("Clusters (K)", k)

st.markdown("---")

if not features:
    st.warning("Please select at least one feature from the sidebar.")
    st.stop()

missing = [f for f in features if f not in df.columns]
if missing:
    st.error(f"Features not found in dataset: {missing}")
    st.stop()

X = df[features].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertias, silhouettes = [], []
K_range = range(2, 10)
for ki in K_range:
    km = KMeans(n_clusters=ki, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df_clean = df.loc[X.index].copy()
df_clean["Cluster"] = kmeans.fit_predict(X_scaled)
sil_score = silhouette_score(X_scaled, df_clean["Cluster"])

COLORS = ["#c9a84c", "#e05252", "#52b0e0", "#7ed87e", "#b07edc", "#e09c52", "#e0d452", "#52e0c4"]
cluster_colors = {i: COLORS[i % len(COLORS)] for i in range(k)}

def generate_insight(cluster_id, stats):
    income = stats.get("Annual Income (k$)", 50)
    score = stats.get("Spending Score (1-100)", 50)
    has_both = "Annual Income (k$)" in stats.index and "Spending Score (1-100)" in stats.index
    if has_both:
        if income > 70 and score > 60:
            return "💎 Premium Loyalists", "Target with exclusive offers, loyalty rewards, VIP programs, and early access to premium product launches. These are your most valuable customers — retain them at all costs."
        elif income > 70 and score <= 60:
            return "💼 Cautious High Earners", "Send ROI-focused content, detailed product comparisons, and trust-building campaigns. Offer money-back guarantees and premium customer support to convert hesitation into purchase."
        elif income <= 70 and score > 60:
            return "🛍️ Enthusiastic Bargain Hunters", "Flash sales, limited-time discount codes, buy-one-get-one offers, and FOMO-driven promotions work best. They love shopping — just need the right price trigger."
        else:
            return "😴 Low-Engagement Segment", "Re-engage with personalized outreach, win-back email campaigns, surveys to understand needs, and small incentives like free shipping or first-purchase discounts."
    return f"Segment {cluster_id}", "Collect more behavioral data for targeted strategy."

cluster_stats = df_clean.groupby("Cluster")[features].mean()

# CLUSTER SUMMARY
if section == "Cluster Summary":
    st.markdown("### 📊 Cluster Summary & Business Insights")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Customers", len(df_clean))
    m2.metric("Clusters (K)", k)
    m3.metric("Silhouette Score", f"{sil_score:.3f}")
    st.markdown("---")
    cols = st.columns(min(k, 4))
    for i in range(k):
        with cols[i % len(cols)]:
            stats = cluster_stats.loc[i]
            label, strategy = generate_insight(i, stats)
            count = len(df_clean[df_clean["Cluster"] == i])
            pct = count / len(df_clean) * 100
            st.markdown(f"**C{i} — {label}**")
            st.markdown(f"👥 **{count}** customers ({pct:.0f}%)")
            if "Annual Income (k$)" in features:
                st.markdown(f"💰 Avg Income: **${stats['Annual Income (k$)']:.0f}k**")
            if "Spending Score (1-100)" in features:
                st.markdown(f"⭐ Avg Score: **{stats['Spending Score (1-100)']:.0f}**")
            if "Age" in features:
                st.markdown(f"🎂 Avg Age: **{stats['Age']:.0f}**")
            st.info(f"📌 {strategy}")

# VISUALIZATIONS
elif section == "Visualizations":
    st.markdown("### 📈 Visualizations")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Cluster Distribution**")
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0a0a0f")
        ax.set_facecolor("#13131a")
        if len(features) >= 2:
            for i in range(k):
                mask = df_clean["Cluster"] == i
                ax.scatter(df_clean.loc[mask, features[0]], df_clean.loc[mask, features[1]],
                           c=cluster_colors[i], s=40, alpha=0.8, label=f"C{i}")
            ax.set_xlabel(features[0], color="#aaa")
            ax.set_ylabel(features[1], color="#aaa")
        else:
            ax.hist(df_clean[features[0]], bins=20, color=COLORS[0])
            ax.set_xlabel(features[0], color="#aaa")
        ax.tick_params(colors="#aaa")
        for sp in ax.spines.values(): sp.set_edgecolor("#2a2a3a")
        ax.legend(facecolor="#0a0a0f", edgecolor="#2a2a3a", labelcolor="#e8e4d9")
        st.pyplot(fig)

    with col2:
        st.markdown("**Elbow Method (Optimal K)**")
        fig2, ax2 = plt.subplots(figsize=(6, 4), facecolor="#0a0a0f")
        ax2.set_facecolor("#13131a")
        ax2.plot(list(K_range), inertias, color="#c9a84c", marker="o", linewidth=2, markersize=6)
        ax2.axvline(x=k, color="#e05252", linestyle="--", linewidth=1.5, label=f"K={k}")
        ax2.set_xlabel("K", color="#aaa")
        ax2.set_ylabel("Inertia", color="#aaa")
        ax2.tick_params(colors="#aaa")
        for sp in ax2.spines.values(): sp.set_edgecolor("#2a2a3a")
        ax2.legend(facecolor="#0a0a0f", edgecolor="#2a2a3a", labelcolor="#e8e4d9")
        st.pyplot(fig2)

    st.markdown("**Silhouette Score vs K**")
    fig4, ax4 = plt.subplots(figsize=(10, 3), facecolor="#0a0a0f")
    ax4.set_facecolor("#13131a")
    ax4.plot(list(K_range), silhouettes, color="#52b0e0", marker="o", linewidth=2, markersize=6)
    ax4.axvline(x=k, color="#e05252", linestyle="--", linewidth=1.5, label=f"K={k}")
    ax4.set_xlabel("K", color="#aaa")
    ax4.set_ylabel("Silhouette Score", color="#aaa")
    ax4.tick_params(colors="#aaa")
    for sp in ax4.spines.values(): sp.set_edgecolor("#2a2a3a")
    ax4.legend(facecolor="#0a0a0f", edgecolor="#2a2a3a", labelcolor="#e8e4d9")
    st.pyplot(fig4)

    if X_scaled.shape[1] >= 2:
        st.markdown("**PCA Visualization (2D)**")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        fig5, ax5 = plt.subplots(figsize=(8, 5), facecolor="#0a0a0f")
        ax5.set_facecolor("#13131a")
        for i in range(k):
            mask = df_clean["Cluster"].values == i
            ax5.scatter(X_pca[mask, 0], X_pca[mask, 1],
                        c=cluster_colors[i], s=50, alpha=0.8, label=f"C{i}")
        ax5.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", color="#aaa")
        ax5.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", color="#aaa")
        ax5.tick_params(colors="#aaa")
        for sp in ax5.spines.values(): sp.set_edgecolor("#2a2a3a")
        ax5.legend(facecolor="#0a0a0f", edgecolor="#2a2a3a", labelcolor="#e8e4d9")
        st.pyplot(fig5)
        st.caption(f"Total variance explained: {sum(pca.explained_variance_ratio_)*100:.1f}%")

    st.markdown("**Cluster Size Distribution**")
    fig3, ax3 = plt.subplots(figsize=(10, 3), facecolor="#0a0a0f")
    ax3.set_facecolor("#13131a")
    sizes = df_clean["Cluster"].value_counts().sort_index()
    bars = ax3.bar([f"Cluster {i}" for i in sizes.index], sizes.values,
                   color=[cluster_colors[i] for i in sizes.index], edgecolor="#0a0a0f")
    for bar, val in zip(bars, sizes.values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 str(val), ha='center', color="#e8e4d9", fontsize=10)
    ax3.tick_params(colors="#aaa")
    for sp in ax3.spines.values(): sp.set_edgecolor("#2a2a3a")
    st.pyplot(fig3)

# ML EXPLANATION
elif section == "ML Explanation":
    st.markdown("### 🔬 ML Explanation")
    st.markdown("#### How K-Means Works")
    st.markdown("""
1. **Initialize** — Randomly place K centroids in the feature space
2. **Assign** — Each customer is assigned to the nearest centroid (Euclidean distance)
3. **Update** — Recalculate centroids as the mean of all assigned points
4. **Repeat** — Steps 2–3 repeat until centroids stop moving (convergence)
    """)
    st.markdown("---")
    st.markdown("#### Evaluation Metrics")
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Inertia (Within-Cluster Variance)", f"{kmeans.inertia_:.2f}")
        st.caption("Lower = tighter clusters. Used in Elbow Method.")
    with m2:
        st.metric("Silhouette Score", f"{sil_score:.3f}")
        st.caption("Range: -1 to 1. Closer to 1 = well-separated clusters.")
    st.markdown("---")
    st.markdown("#### Limitations of K-Means")
    st.warning("""
- Sensitive to initial centroid placement
- Requires predefined K
- Not suitable for non-spherical clusters
- Outliers can distort cluster centers
    """)
    st.markdown("#### Future Enhancements")
    st.success("""
- DBSCAN — handles non-spherical clusters, detects outliers automatically
- Hierarchical Clustering — no need to predefine K
- RFM Segmentation — Recency, Frequency, Monetary value
- Real-time data integration with CRM systems
- Predict segment for new customers
    """)

# PREDICT NEW CUSTOMER
elif section == "Predict New Customer":
    st.markdown("### 🤖 Predict Segment for a New Customer")
    st.markdown("Enter a new customer's details:")
    input_vals = {}
    cols = st.columns(len(features))
    for idx, feat in enumerate(features):
        with cols[idx]:
            if feat == "Age":
                input_vals[feat] = st.number_input("Age", min_value=18, max_value=100, value=30)
            elif feat == "Annual Income (k$)":
                input_vals[feat] = st.number_input("Annual Income (k$)", min_value=10, max_value=200, value=60)
            elif feat == "Spending Score (1-100)":
                input_vals[feat] = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)

    if st.button("Predict Segment"):
        new_data = np.array([[input_vals[f] for f in features]])
        new_scaled = scaler.transform(new_data)
        pred_cluster = kmeans.predict(new_scaled)[0]
        stats = cluster_stats.loc[pred_cluster]
        label, strategy = generate_insight(pred_cluster, stats)
        st.success(f"### This customer belongs to: C{pred_cluster} — {label}")
        st.markdown(f"📌 **Strategy:** {strategy}")
        if len(features) >= 2:
            fig, ax = plt.subplots(figsize=(7, 4), facecolor="#0a0a0f")
            ax.set_facecolor("#13131a")
            for i in range(k):
                mask = df_clean["Cluster"] == i
                ax.scatter(df_clean.loc[mask, features[0]], df_clean.loc[mask, features[1]],
                           c=cluster_colors[i], s=30, alpha=0.5, label=f"C{i}")
            ax.scatter(input_vals[features[0]], input_vals[features[1]],
                       c="white", s=200, marker="*", zorder=5, label="New Customer")
            ax.set_xlabel(features[0], color="#aaa")
            ax.set_ylabel(features[1], color="#aaa")
            ax.tick_params(colors="#aaa")
            for sp in ax.spines.values(): sp.set_edgecolor("#2a2a3a")
            ax.legend(facecolor="#0a0a0f", edgecolor="#2a2a3a", labelcolor="#e8e4d9")
            st.pyplot(fig)

# ALGORITHM COMPARISON
elif section == "Algorithm Comparison":
    st.markdown("### ⚖️ Multi-Algorithm Comparison")
    km_labels = kmeans.labels_
    km_sil = silhouette_score(X_scaled, km_labels)
    agg = AgglomerativeClustering(n_clusters=k)
    agg_labels = agg.fit_predict(X_scaled)
    agg_sil = silhouette_score(X_scaled, agg_labels)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### K-Means")
        st.metric("Silhouette Score", f"{km_sil:.3f}")
        st.metric("Inertia", f"{kmeans.inertia_:.2f}")
        st.markdown("✅ Fast and scalable\n\n✅ Simple to understand\n\n❌ Needs K predefined\n\n❌ Sensitive to outliers")
    with col2:
        st.markdown("#### Agglomerative (Hierarchical)")
        st.metric("Silhouette Score", f"{agg_sil:.3f}")
        st.markdown("✅ No need to predefine K\n\n✅ Works with any shape\n\n❌ Slower on large data\n\n❌ Cannot undo merges")

    if X_scaled.shape[1] >= 2:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#0a0a0f")
        for ax, title, lbls in zip(axes, ["K-Means", "Agglomerative"], [km_labels, agg_labels]):
            ax.set_facecolor("#13131a")
            for i in range(k):
                mask = lbls == i
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=COLORS[i % len(COLORS)], s=30, alpha=0.7, label=f"C{i}")
            ax.set_title(title, color="#e8e4d9")
            ax.tick_params(colors="#aaa")
            for sp in ax.spines.values(): sp.set_edgecolor("#2a2a3a")
            ax.legend(facecolor="#0a0a0f", edgecolor="#2a2a3a", labelcolor="#e8e4d9", fontsize=7)
        fig.patch.set_facecolor("#0a0a0f")
        st.pyplot(fig)

    winner = "K-Means" if km_sil >= agg_sil else "Agglomerative"
    st.info(f"🏆 Better Silhouette Score: **{winner}**")

# EXPORT
elif section == "Export":
    st.markdown("### 💾 Export Results")
    csv = df_clean.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Segmented Data (CSV)", data=csv,
                       file_name="customer_segments.csv", mime="text/csv")
    st.markdown("---")
    st.markdown(f"- Total customers: **{len(df_clean)}**")
    st.markdown(f"- Clusters: **{k}**")
    st.markdown(f"- Silhouette Score: **{sil_score:.3f}**")
    st.markdown(f"- Inertia: **{kmeans.inertia_:.2f}**")
    for i in range(k):
        stats = cluster_stats.loc[i]
        label, _ = generate_insight(i, stats)
        count = len(df_clean[df_clean["Cluster"] == i])
        st.markdown(f"- **C{i} ({label}):** {count} customers")

st.markdown("---")
st.caption("Built with K-Means · PCA · Silhouette · Scikit-learn · Streamlit | BTech AI/ML Project")
