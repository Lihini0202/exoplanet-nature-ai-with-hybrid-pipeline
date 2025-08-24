# Kepler Exoplanet Analysis with Hybrid Pipeline 🌌

Welcome to the **Kepler Exoplanet Analysis with Hybrid Pipeline**, an interactive web application built with Streamlit to explore and visualize exoplanet data using advanced machine learning models. This project combines the power of Ant Colony Optimization (ACO), Cuckoo Search (CS), Whale Optimization Algorithm (WOA), and Convolutional Neural Networks (CNN) to classify exoplanets and uncover their habitability potential. Dive into a universe of data with stunning visualizations and real-time insights!

---

## ✨ Features

- **Interactive Interface**: Explore 5 dynamic tabs—Models, Robustness, Habitability, Visualizations, and Conclusion.
- **Model Performance**: Evaluate ACO, CS, WOA, and CNN models with pre-computed and live F1-scores.
- **Live Visualizations**: Enjoy animated scatter plots, ROC curves, confusion matrices, learning curves, and SHAP feature importance plots.
- **Habitability Insights**: Adjust thresholds for equilibrium temperature and insolation flux to discover potentially habitable exoplanets.
- **Dataset Overview**: Access summary statistics and a preview of the Kepler exoplanet dataset.
- **Modern Design**: Built with Plotly for GIF-like interactivity and Seaborn for elegant static plots.

---

## 🚀 Demo

Check out the live app https://kvqcvtpg9sqjnvlhydl42x.streamlit.app/ 

---

## 📊 Dataset

This project uses the **Kepler Exoplanet Dataset** from Kaggle, containing 9,564 exoplanet candidates with 49 features, including:
- `koi_disposition`: Classification (CONFIRMED, FALSE POSITIVE, CANDIDATE).
- `koi_period`, `koi_teq`, `koi_insol`: Orbital period, equilibrium temperature, and insolation flux.
- Error metrics and stellar properties.

Dataset source: [Kaggle - Exoplanets](https://www.kaggle.com/datasets/arashnic/exoplanets).

---

## 🛠️ Installation

To run this app locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Lihini0202/exoplanet-nature-ai-with-hybrid-pipeline.git
   cd exoplanet-nature-ai-with-hybrid-pipeline

Install Dependencies:
Ensure Python 3.8+ is ready, then:
bashpip install -r requirements.txt

Prepare the Galaxy:

Place exoplanets.csv (from Kaggle) in the project root.
Add cs_model.joblib (pre-trained CS model) to the root.


Ignite the App:
bashstreamlit run app.py
Visit http://localhost:8501 to explore the universe!


## 🛡️ Requirements

**Libraries:**  
`streamlit`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `joblib`, `tensorflow`, `shap`, `plotly`

**Files:**  
`exoplanets.csv`, `cs_model.joblib`, `requirements.txt`, `app.py`

**Update `requirements.txt` with:**  
streamlit==1.30.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.0
xgboost==2.1.1
matplotlib==3.9.0
seaborn==0.13.2
joblib==1.4.2
tensorflow==2.17.0
shap==0.44.1
plotly==5.18.0


---

## 🎨 Exploring the App: Tab Adventures

- **Models**: Check F1-scores for all models, with **CS updated live**.  
- **Robustness**: Compare robustness metrics (e.g., CS: 0.6981 ± 0.0035).  
- **Habitability**: Watch a **live scatter plot** evolve with your slider tweaks.  
- **Visualizations**: Marvel at **confusion matrices, ROC curves, learning curves, and SHAP plots**.  
- **Conclusion**: Reflect on findings and plan your **future cosmic quests**.  

---

## 🌌 Interactive Galaxies

- Slide to adjust **habitability thresholds** and click **"Refresh Habitability Plot"** for a new perspective.  
- Hover over **Plotly charts** to zoom in and explore the data cosmos in detail.  

---

## 🌠 Future Horizons

- Integrate **real-time TESS data**  
- Animate **learning curves** with training history  
- Expand **SHAP analysis** across all models
