# 🧠 Customer Segmentation AI
### BTech AI/ML Project — K-Means Clustering Dashboard

---

## 📁 Files
- `app.py` — Main Streamlit application
- `requirements.txt` — Python dependencies

---

## 🚀 How to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run the app
```bash
streamlit run app.py
```

### Step 3 — Open in browser
Streamlit will auto-open at `http://localhost:8501`

---

## 🎯 Features
- ✅ Sample dataset (200 customers) built-in
- ✅ Upload your own CSV
- ✅ Adjustable K (number of clusters)
- ✅ Elbow Method graph to justify K
- ✅ Cluster scatter plot visualization
- ✅ Business insight per cluster
- ✅ Download segmented data as CSV

---

## 🧪 Tech Stack
| Tool | Purpose |
|------|---------|
| Python | Core language |
| Scikit-learn | K-Means algorithm |
| Pandas / NumPy | Data handling |
| Matplotlib / Seaborn | Visualizations |
| Streamlit | Web dashboard UI |

---

## 📊 How K-Means Works (for viva)
1. Choose K clusters
2. Randomly place K centroids
3. Assign each point to nearest centroid
4. Recalculate centroids
5. Repeat until convergence

**Elbow Method** — plot inertia vs K, pick the "elbow" point where reduction slows down.

---

## 💡 Business Insights Generated
| Cluster Type | Strategy |
|---|---|
| 💎 Premium Loyalists | Exclusive offers, loyalty rewards |
| 💼 Cautious High Earners | ROI-focused, trust-building |
| 🛍️ Enthusiastic Bargain Hunters | Flash sales, FOMO campaigns |
| 😴 Low-Engagement | Win-back campaigns, surveys |
