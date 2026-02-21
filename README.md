<div id="top">

<div align="center">

# ML-EXPERIMENTS

<em>A portfolio of applied Machine Learning, Deep Learning, and NLP experiments, built with Python and Jupyter.</em>

<a href="https://github.com/omarg-dev/ml-experiments/commits/master"><img src="https://img.shields.io/github/last-commit/omarg-dev/ml-experiments?style=flat&logo=git&logoColor=white&color=0080ff" alt="last-commit"></a>
<a href="https://github.com/omarg-dev/ml-experiments"><img src="https://img.shields.io/github/languages/top/omarg-dev/ml-experiments?style=flat&color=0080ff" alt="repo-top-language"></a>
<a href="https://github.com/omarg-dev/ml-experiments"><img src="https://img.shields.io/github/languages/count/omarg-dev/ml-experiments?style=flat&color=0080ff" alt="repo-language-count"></a>

<em>Built with:</em>

<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/Jupyter-F37626.svg?style=flat&logo=Jupyter&logoColor=white" alt="Jupyter">
<img src="https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=flat&logo=Streamlit&logoColor=white" alt="Streamlit">

</div>
<br>

---

## 📄 Table of Contents

- [✨ Overview](#-overview)
- [📌 Features](#-features)
- [📁 Project Structure](#-project-structure)
- [📑 Project Index](#-project-index)

---

## ✨ Overview

`ml-experiments` is a portfolio and experimentation hub for applied Machine Learning, Deep Learning, and Natural Language Processing. Built with Python and Jupyter Notebooks, it covers end-to-end pipelines from exploratory analysis through model training and evaluation across a diverse range of real-world domains.

Use cases include pneumonia detection from chest X-rays (CNN), binary sentiment classification on Amazon reviews (LSTM), unsupervised e-commerce customer segmentation, disaster tweet classification with TF-IDF and ensemble classifiers, National ID field extraction via OCR, and harmful prompt detection benchmarked against Google Gemini models. A companion Streamlit application (`EDA_data_analysis_streamlit.py`) packages shared preprocessing and modelling utilities into an interactive AutoML interface.

---

## 📌 Features

| Component | Details |
| :--- | :--- |
| 🧠 **Deep Learning Architectures** | CNN for binary pneumonia detection from chest X-rays; stacked LSTM achieving ~90% accuracy on Amazon review sentiment classification. |
| 📊 **Unsupervised Learning** | K-Means clustering and ANN-based pipelines that segment e-commerce customers into High Frequent, Moderate, and Low Infrequent buyer groups. |
| 📝 **Natural Language Processing** | TF-IDF feature extraction with multi-model training (Logistic Regression, SVM, XGBoost, Random Forest) for disaster tweet classification; LLM-based harmful prompt detection benchmarked against Google Gemini Flash and Pro via LangChain. |
| 👁️ **Computer Vision & OCR** | End-to-end pipeline for ingesting National ID card images and extracting structured identity fields (name, date of birth, ID number) via OCR. |
| 🛠️ **Automated Data Prep** | `DataPrepKit` notebook providing reusable preprocessing: CSV, Excel, and JSON ingestion; multi-strategy missing value imputation; one-hot and label encoding. |
| 📉 **Exploratory Data Analysis** | Statistical analysis, outlier removal, pricing trends, and neighbourhood demand visualisation on Dublin Airbnb listings, with a stakeholder-facing business recommendations report. |
| 🌐 **Streamlit AutoML App** | Interactive web application consolidating file upload, EDA visualisation, preprocessing, and automated or manual model selection with live performance metrics. |

---

## 📁 Project Structure

```sh
└── ml-experiments/
    ├── EDA_data_analysis_streamlit.py
    ├── README.md
    ├── ANN_market_segmentation.ipynb
    ├── CNN_medical_image_analysis.ipynb
    ├── EDA_airbnb_listing_analysis.ipynb
    ├── LCF_harmful_prompt_detection.ipynb
    ├── MISC_data_prep_kit.ipynb
    ├── NLP_twitter_disaster.ipynb
    ├── OCR_national_id_recognition.ipynb
    ├── RNN_sentiment_analysis_with_lstm.ipynb
    └── UL_customer_segmentation.ipynb
```

---

## 📑 Project Index

<details open>
<summary><b>ML-EXPERIMENTS/</b></summary>

| File | Summary |
| :--- | :--- |
| [EDA_data_analysis_streamlit.py](https://github.com/omarg-dev/ml-experiments/blob/master/EDA_data_analysis_streamlit.py) | Streamlit-based AutoML web application. Consolidates file upload, EDA visualisation, null value treatment, categorical encoding, and automated or manual model selection into a single interactive interface with live performance metrics. |
| [MISC_data_prep_kit.ipynb](https://github.com/omarg-dev/ml-experiments/blob/master/MISC_data_prep_kit.ipynb) | Reusable preprocessing utility supporting CSV, Excel, and JSON ingestion. Provides dataset summarisation, multi-strategy missing value imputation, and one-hot/label encoding to produce model-ready datasets. |
| [RNN_sentiment_analysis_with_lstm.ipynb](https://github.com/omarg-dev/ml-experiments/blob/master/RNN_sentiment_analysis_with_lstm.ipynb) | Binary sentiment classification on Amazon product reviews using a stacked LSTM. Covers tokenisation, sequence padding, model training, and evaluation; achieves ~90% test accuracy. |
| [CNN_medical_image_analysis.ipynb](https://github.com/omarg-dev/ml-experiments/blob/master/CNN_medical_image_analysis.ipynb) | Binary pneumonia detection from chest X-ray scans. Implements a CNN with data augmentation, evaluates results via classification report and confusion matrix, and persists the trained model for downstream use. |
| [UL_customer_segmentation.ipynb](https://github.com/omarg-dev/ml-experiments/blob/master/UL_customer_segmentation.ipynb) | Unsupervised customer segmentation on online retail transaction data. Applies RFM feature engineering and clustering to produce three behavioural tiers: High Frequent, Moderate, and Low Infrequent buyers. |
| [NLP_twitter_disaster.ipynb](https://github.com/omarg-dev/ml-experiments/blob/master/NLP_twitter_disaster.ipynb) | Disaster tweet classification using TF-IDF features and four classifiers (Logistic Regression, SVM, XGBoost, Random Forest). Benchmarks all models to identify the best fit for real-time emergency monitoring. |
| [EDA_airbnb_listing_analysis.ipynb](https://github.com/omarg-dev/ml-experiments/blob/master/EDA_airbnb_listing_analysis.ipynb) | EDA on Dublin Airbnb listings covering data cleaning, outlier removal, pricing trends, neighbourhood demand, room type distributions, and host ownership patterns. Concludes with a stakeholder business plan for the short-term rental market. |
| [OCR_national_id_recognition.ipynb](https://github.com/omarg-dev/ml-experiments/blob/master/OCR_national_id_recognition.ipynb) | OCR pipeline for National ID documents. Ingests card images, applies text detection and recognition, and structures extracted fields (name, date of birth, ID number) into machine-readable output. |
| [LCF_harmful_prompt_detection.ipynb](https://github.com/omarg-dev/ml-experiments/blob/master/LCF_harmful_prompt_detection.ipynb) | Benchmarks Google Gemini Flash and Pro for binary harmful prompt detection using a LangChain classification pipeline on the LLM Evaluation Hub dataset. Reports accuracy against ground-truth labels via standard classification metrics. |
| [ANN_market_segmentation.ipynb](https://github.com/omarg-dev/ml-experiments/blob/master/ANN_market_segmentation.ipynb) | ANN-based customer market segmentation on e-commerce transactional data (`e-commerce_data.csv`). Complements the unsupervised clustering notebook with a neural network approach to identifying distinct customer groups for targeted marketing. |

</details>

---
