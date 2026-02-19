<div id="top">

<!-- HEADER STYLE: CLASSIC -->
<div align="center">

# ML-EXPERIMENTS.GIT

<em></em>

<!-- BADGES -->
<img src="https://img.shields.io/github/license/omarg-dev/ml-experiments?style=flat&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
<img src="https://img.shields.io/github/last-commit/omarg-dev/ml-experiments?style=flat&logo=git&logoColor=white&color=0080ff" alt="last-commit">
<img src="https://img.shields.io/github/languages/top/omarg-dev/ml-experiments?style=flat&color=0080ff" alt="repo-top-language">
<img src="https://img.shields.io/github/languages/count/omarg-dev/ml-experiments?style=flat&color=0080ff" alt="repo-language-count">

<em>Built with the tools and technologies:</em>

<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">

</div>
<br>

---

## 📄 Table of Contents

- [📄 Table of Contents](#-table-of-contents)
- [✨ Overview](#-overview)
- [📌 Features](#-features)
- [📁 Project Structure](#-project-structure)
  - [📑 Project Index](#-project-index)
- [📜 License](#-license)

---

## ✨ Overview

The `ml-experiments` repository serves as a structured portfolio and experimentation hub for applied Machine Learning, Deep Learning, and Natural Language Processing. Built primarily using Python and Jupyter Notebooks, the project demonstrates end-to-end data pipelines ranging from exploratory data analysis (EDA) to the deployment-ready training of complex neural networks. 

With its experimental nature, the repository contains a diverse set of real-world use cases, including medical image classification (CNNs), social media sentiment analysis (LSTMs), unsupervised e-commerce customer segmentation, and harmful prompt detection using Large Language Models (LLMs). It acts as both a personal reference library for architectural patterns and a showcase of practical, data-driven problem solving.

---

## 📌 Features

| Component | Details |
| :--- | :--- |
| 🧠 **Deep Learning Architectures** | Implements Convolutional Neural Networks (CNNs) for medical image analysis and Recurrent Neural Networks (LSTMs) for sequential text classification. |
| 📊 **Unsupervised Learning** | Applies clustering algorithms and Artificial Neural Networks (ANNs) to segment e-commerce customers based on purchasing behavior. |
| 📝 **Natural Language Processing** | Features end-to-end NLP pipelines, including TF-IDF feature extraction for disaster tweet classification and LLM-driven evaluations for harmful prompt detection. |
| 👁️ **Computer Vision & OCR** | Includes targeted workflows for extracting and structuring data from National ID documents using Optical Character Recognition. |
| 🛠️ **Automated Data Prep** | Contains reusable utility scripts (`DataPrepKit`) and helper libraries to streamline data ingestion, missing value imputation, and categorical encoding. |
| 📉 **Exploratory Data Analysis** | Demonstrates rigorous statistical analysis, outlier removal, and trend visualization (e.g., Airbnb market analysis) to drive business recommendations. |

---

## 📁 Project Structure

```sh
└── ml-experiments.git/
    ├── ML helper lib.py
    ├── README.md
    ├── [ANN] Market Segmentation with Neural Networks.ipynb
    ├── [CNN] Medical Image Analysis with CNN.ipynb
    ├── [EDA] Airbnb Listing Analysis.ipynb
    ├── [LCF] Harmful Prompt Detection.ipynb
    ├── [Misc] DataPrepKit.ipynb
    ├── [NLP] Twitter Disaster.ipynb
    ├── [OCR] National ID Recognition.ipynb
    ├── [RNN] Sentiment Analysis on Social Media with LSTM.ipynb
    └── [UL] Customer Segmentation for E-commerce Personalization.ipynb
```

### 📑 Project Index

<details open>
        <summary><b><code>ML-EXPERIMENTS.GIT/</code></b></summary>
                <blockquote>
                    <div class='directory-path' style='padding: 8px 0; color: #666;'>
                            <code><b>⦿ __root__</b></code>
                    <table style='width: 100%; border-collapse: collapse;'>
                    <thead>
                            <tr style='background-color: #f8f9fa;'>
                                    <th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
                                    <th style='text-align: left; padding: 8px;'>Summary</th>
                            </tr>
                    </thead>
                            <tr style='border-bottom: 1px solid #eee;'>
                                    <td style='padding: 8px;'><b><a href='https://github.com/omarg-dev/ml-experiments.git/blob/master/ML helper lib.py'>ML helper lib.py</a></b></td>
                                    <td style='padding: 8px;'>- Serves as the core backbone of a Streamlit-based AutoML web application, consolidating data ingestion, exploratory analysis, preprocessing, and model training into a unified helper library<br>- It enables end-to-end machine learning workflows by handling file uploads, visualization, null value treatment, categorical encoding, and automated or manual model selection, exposing results and performance metrics directly within the interactive user interface.</td>
                            </tr>
                            <tr style='border-bottom: 1px solid #eee;'>
                                    <td style='padding: 8px;'><b><a href='https://github.com/omarg-dev/ml-experiments.git/blob/master/[Misc] DataPrepKit.ipynb'>[Misc] DataPrepKit.ipynb</a></b></td>
                                    <td style='padding: 8px;'>- A reusable data preparation utility encapsulated within a Jupyter Notebook, serving as a foundational preprocessing layer for machine learning workflows<br>- Designed to streamline ingestion of CSV, Excel, and JSON datasets, it provides summarization, missing value handling through multiple imputation strategies, and categorical encoding via one-hot or label encoding — enabling clean, model-ready datasets before any downstream training or analysis begins.</td>
                            </tr>
                            <tr style='border-bottom: 1px solid #eee;'>
                                    <td style='padding: 8px;'><b><a href='https://github.com/omarg-dev/ml-experiments.git/blob/master/[RNN] Sentiment Analysis on Social Media with LSTM.ipynb'>[RNN] Sentiment Analysis on Social Media with LSTM.ipynb</a></b></td>
                                    <td style='padding: 8px;'>- Demonstrates end-to-end binary sentiment classification on Amazon product reviews using a stacked LSTM neural network<br>- Covering data ingestion, text tokenization, model training, and performance evaluation, it serves as a practical RNN-focused reference within the broader machine learning notebook collection<br>- The model achieves ~90% test accuracy, providing a strong baseline for social media and review-based sentiment analysis tasks.</td>
                            </tr>
                            <tr style='border-bottom: 1px solid #eee;'>
                                    <td style='padding: 8px;'><b><a href='https://github.com/omarg-dev/ml-experiments.git/blob/master/[CNN] Medical Image Analysis with CNN.ipynb'>[CNN] Medical Image Analysis with CNN.ipynb</a></b></td>
                                    <td style='padding: 8px;'>- Demonstrates an end-to-end pipeline for binary medical image classification, specifically detecting pneumonia from chest X-ray scans<br>- Covering data preprocessing, CNN model construction, training, and evaluation, the notebook serves as a foundational deep learning workflow within the codebase<br>- Results are assessed using classification metrics and confusion matrices, with the trained model persisted for potential downstream or clinical application use.</td>
                            </tr>
                            <tr style='border-bottom: 1px solid #eee;'>
                                    <td style='padding: 8px;'><b><a href='https://github.com/omarg-dev/ml-experiments.git/blob/master/[UL] Customer Segmentation for E-commerce Personalization.ipynb'>[UL] Customer Segmentation for E-commerce Personalization.ipynb</a></b></td>
                                    <td style='padding: 8px;'>- Implements an end-to-end unsupervised learning pipeline for e-commerce customer segmentation using online retail transaction data<br>- Customers are grouped into distinct behavioral segments — High Frequent, Moderate, and Low Infrequent buyers — enabling targeted marketing strategies tailored to each group<br>- Serves as the core analytical notebook driving personalization decisions across the broader e-commerce recommendation and customer retention architecture.</td>
                            </tr>
                            <tr style='border-bottom: 1px solid #eee;'>
                                    <td style='padding: 8px;'><b><a href='https://github.com/omarg-dev/ml-experiments.git/blob/master/[NLP] Twitter Disaster.ipynb'>[NLP] Twitter Disaster.ipynb</a></b></td>
                                    <td style='padding: 8px;'>- An end-to-end NLP pipeline that classifies tweets as real disaster reports or non-disaster content<br>- Sourced from a Kaggle dataset, the notebook covers data preprocessing, TF-IDF feature extraction, multi-model training (Logistic Regression, SVM, XGBoost, Random Forest), and performance evaluation to identify the best-performing classifier, with practical applications in real-time disaster monitoring and emergency response coordination.</td>
                            </tr>
                            <tr style='border-bottom: 1px solid #eee;'>
                                    <td style='padding: 8px;'><b><a href='https://github.com/omarg-dev/ml-experiments.git/blob/master/[EDA] Airbnb Listing Analysis.ipynb'>[EDA] Airbnb Listing Analysis.ipynb</a></b></td>
                                    <td style='padding: 8px;'>- Exploratory Data Analysis (EDA) notebook examining Airbnb listing data for Dublin, Ireland, serving as the primary analytical layer of the project<br>- Covers data cleaning, outlier removal, pricing trends, neighbourhood demand, room type distributions, and host ownership patterns, culminating in a stakeholder-facing business plan with strategic recommendations for navigating the Dublin short-term rental market.</td>
                            </tr>
                            <tr style='border-bottom: 1px solid #eee;'>
                                    <td style='padding: 8px;'><b><a href='https://github.com/omarg-dev/ml-experiments.git/blob/master/[OCR] National ID Recognition.ipynb'>[OCR] National ID Recognition.ipynb</a></b></td>
                                    <td style='padding: 8px;'>- Summary<strong>National ID Recognition Notebook</strong> (<code>[OCR] National ID Recognition.ipynb</code>)This Jupyter Notebook serves as the <strong>core OCR (Optical Character Recognition) pipeline</strong> for extracting and interpreting structured data from National ID documents<br>- It acts as the primary experimentation and implementation hub within the codebase for automating identity document processing.### Main PurposeThe notebook orchestrates the end-to-end workflow of ingesting National ID card images, applying OCR techniques to detect and extract key identity fields (such as name, date of birth, ID number, etc.), and structuring the recognized text into usable, machine-readable output.### Role in the Architecture-Serves as the <strong>entry point and reference implementation</strong> for the document recognition pipeline-Bridges raw image input with structured identity data extraction-Likely underpins downstream components such as data validation, storage, or API integration within the broader system-Functions as both a <strong>prototyping environment</strong> and <strong>documented workflow</strong> for the OCR process, making it accessible for iterative improvement and onboarding### Key ValueThis notebook centralizes the intelligence for automating what would otherwise be a manual identity verification process, enabling scalable, consistent, and repeatable extraction of information from National ID documents across the system.</td>
                            </tr>
                            <tr style='border-bottom: 1px solid #eee;'>
                                    <td style='padding: 8px;'><b><a href='https://github.com/omarg-dev/ml-experiments.git/blob/master/[LCF] Harmful Prompt Detection.ipynb'>[LCF] Harmful Prompt Detection.ipynb</a></b></td>
                                    <td style='padding: 8px;'>- Evaluates the capability of Google Gemini models (Flash and Pro) in detecting harmful prompts using a LangChain-powered classification pipeline<br>- Operating on the LLM Evaluation Hub dataset, it preprocesses and samples text data, queries both models with structured prompts, and benchmarks their binary harm-detection accuracy against ground truth labels using standard classification metrics.</td>
                            </tr>
                            <tr style='border-bottom: 1px solid #eee;'>
                                    <td style='padding: 8px;'><b><a href='https://github.com/omarg-dev/ml-experiments.git/blob/master/[ANN] Market Segmentation with Neural Networks.ipynb'>[ANN] Market Segmentation with Neural Networks.ipynb</a></b></td>
                                    <td style='padding: 8px;'>- Market Segmentation with Neural Networks## SummaryThis Jupyter Notebook serves as the <strong>primary analytical workbook</strong> for performing customer market segmentation using Artificial Neural Networks (ANN) on e-commerce data<br>- It represents the core machine learning pipeline within the project, sitting alongside other analytical notebooks in the codebase.## PurposeThe notebook drives the end-to-end workflow of segmenting an e-commerce customer base into meaningful groups by leveraging neural network models<br>- It consumes raw transactional data from the shared <code>Datasets</code> directory, processes it, and applies ANN-based techniques to uncover distinct customer segments that can inform targeted marketing strategies and business decisions.## Role in the Codebase-Acts as the <strong>neural network-focused segmentation solution</strong>, likely complementing other segmentation approaches (e.g., traditional clustering methods) present elsewhere in the project-Consumes the <strong>centralized e-commerce dataset</strong> (<code>e-commerce_data.csv</code>) shared across the broader codebase-Represents the <strong>ANN-specific analytical layer</strong>, distinguishing itself from other notebooks through its deep learning approach to customer classification-Serves as a <strong>self-contained, reproducible analysis</strong> that data scientists or analysts can run independently to generate customer segment insights## Business ValueBy segmenting customers through neural networks, this notebook enables stakeholders to identify behavioral patterns and customer groups at scale — supporting personalization, retention strategies, and revenue optimization across the e-commerce platform.</td>
                            </tr>
                    </table>
            </blockquote>
</details>

---

## 📜 License

Ml-experiments.git is protected under the [LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

[back-to-top]: https://img.shields.io/badge/-BACK_TO_TOP-151515?style=flat-square

---
