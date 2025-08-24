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
   git clone https://github.com/your-username/exoplanet-nature-ai-with-hybrid-pipeline.git
   cd exoplanet-nature-ai-with-hybrid-pipeline
