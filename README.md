# 🧠 Customer Segmentation AI
### BTech AI/ML Project — K-Means Clustering Business Intelligence Dashboard

🔗 **Live App:** [customer-segmentation-5jqtbhtm84ru4swjqkcxws.streamlit.app](https://customer-segmentation-5jqtbhtm84ru4swjqkcxws.streamlit.app)

---

## 📌 Project Overview
An interactive AI-powered Customer Segmentation dashboard that groups customers by behavior using K-Means Clustering and generates actionable business insights for each segment.

---

## 🚀 Features
| Feature | Description |
|---|---|
| 📊 Cluster Summary | Business insights per cluster with customer stats |
| 📈 Visualizations | Scatter plot, Elbow Method, Silhouette Score, PCA 2D |
| 🔬 ML Explanation | How K-Means works, evaluation metrics, limitations, future scope |
| 🤖 Predict New Customer | Enter details → get segment + position on chart |
| ⚖️ Algorithm Comparison | K-Means vs Agglomerative (Hierarchical) Clustering |
| 💾 Export | Download segmented data as CSV |

---

## 🧪 Tech Stack
| Tool | Purpose |
|---|---|
| Python | Core language |
| Scikit-learn | K-Means, Agglomerative, PCA, Silhouette Score |
| Pandas / NumPy | Data handling |
| Matplotlib | Visualizations |
| Streamlit | Web dashboard UI |

---

## 📊 Evaluation Metrics
- **Inertia** — Within-cluster variance. Lower = tighter clusters. Used in Elbow Method.
- **Silhouette Score** — Range -1 to 1. Closer to 1 = well-separated clusters.

---

## 💡 Cluster Types (Business Segments)
| Cluster | Type | Strategy |
|---|---|---|
| 💎 Premium Loyalists | High income + High spend | Exclusive offers, VIP programs |
| 💼 Cautious High Earners | High income + Low spend | ROI-focused, trust-building |
| 🛍️ Enthusiastic Bargain Hunters | Low income + High spend | Flash sales, FOMO campaigns |
| 😴 Low-Engagement Segment | Low income + Low spend | Win-back campaigns, surveys |

---

## ⚠️ Limitations of K-Means
- Sensitive to initial centroid placement
- Requires predefined K
- Not suitable for non-spherical clusters
- Outliers can distort cluster centers

---

## 🔮 Future Enhancements
- DBSCAN — handles non-spherical clusters, detects outliers
- Hierarchical Clustering — no need to predefine K
- RFM Segmentation — Recency, Frequency, Monetary value
- Real-time CRM data integration

---

## 🏃 How to Run Locally
```bash
git clone https://github.com/konduruyuvashree/customer-segmentation.git
cd customer-segmentation
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 File Structure
```
customer-segmentation/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

*Built with K-Means · PCA · Silhouette · Scikit-learn · Streamlit | BTech AI/ML Project*
