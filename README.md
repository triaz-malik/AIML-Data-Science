# Data Science & AI/ML Project Portfolio

A collection of end-to-end **machine learning, deep learning, AI and data analytics** projects.
Every project folder is self-contained: source code, pipeline scripts / notebooks, and a
**final report (PDF / Word) at the top of the folder** that walks through the business problem,
data, EDA, modeling, results, and impact.

> **Note on artifacts:** large raw datasets, trained model weights, and image corpora are
> intentionally excluded from this repository (see [`.gitignore`](.gitignore)). The report PDFs
> are self-contained — every figure is embedded. To reproduce results, clone a project and
> follow its `requirements.txt`.

---

## 🤖 AIML — Deep Learning · Computer Vision · NLP · LLM Agents

| Project | What it does | Final report |
|---|---|---|
| [Amazon Product Sentiment Analysis](AIML/Amazon%20Product%20Sentiment%20Analysis) | Aspect-based sentiment analysis on Amazon product reviews, with an interactive dashboard. | [📄 PDF](AIML/Amazon%20Product%20Sentiment%20Analysis/Project_Report.pdf) |
| [Amazon Reviews Sentiment](AIML/Amazon%20Reviews%20Sentiment) | 3-class review sentiment on 235K reviews — TF-IDF + LogReg baseline vs. DistilBERT transformer. | [📄 PDF](AIML/Amazon%20Reviews%20Sentiment/Amazon_Reviews_Sentiment_Report.pdf) |
| [Car Damage Assessment](AIML/Car%20Demage) | CNN vehicle-damage detection & severity scoring with Grad-CAM explainability. | [📄 PDF](AIML/Car%20Demage/Vehicle_Damage_Assessment_Report.pdf) |
| [Chest X-Ray — Pneumonia Detection](AIML/Chest%20xRAY) | Pediatric pneumonia screening from chest X-rays (Custom CNN vs. ResNet50), recall-first KPI. | [📄 PDF](AIML/Chest%20xRAY/Pneumonia_Detection_Report.pdf) |
| [Consumer Complaints Classification](AIML/Consumer%20Complaints) | Multi-class text classification routing consumer complaints to the right product category. | [📄 PDF](AIML/Consumer%20Complaints/Consumer_Complaint_Classification_Report.pdf) |
| [Crew AI Programming Team](AIML/Crew%20AI%20Programming%20Team) | Multi-agent (CrewAI) autonomous programming team that plans, codes, and reviews. | [📄 PDF](AIML/Crew%20AI%20Programming%20Team/Crew_AI_Programming_Team_Report.pdf) |
| [Plant Disease Detection](AIML/Plant%20Disease) | Leaf-disease classification across 27 classes / 9 crops (CNN · ResNet50 · EfficientNetB0) + Grad-CAM. | [📄 PDF](AIML/Plant%20Disease/Plant_Disease_Detection_Report.pdf) |
| [SMS Spam Detection](AIML/SMS%20Spam%20Detection) | Spam-vs-ham SMS classification with a lightweight, deployable model. | [📄 PDF](AIML/SMS%20Spam%20Detection/Project_Report.pdf) |
| [Steel Plant Manufacturing Fault](AIML/Steel%20Plant%20Manufacturing%20Fault) | Severstal steel surface-defect detection — multi-label classification + U-Net segmentation + severity scoring. | [📄 PDF](AIML/Steel%20Plant%20Manufacturing%20Fault/Steel_Defect_Detection_Report.pdf) |

## 📊 Data Science — Tabular ML · Analytics · Forecasting · Segmentation

| Project | What it does | Final report |
|---|---|---|
| [Credit Card Fraud Detection](Data-Science/Credit%20Card%20%20Fraud%20Detection) | Imbalanced fraud detection (PR-AUC focus, XGBoost) with SHAP explanations and threshold tuning. | [📄 PDF](Data-Science/Credit%20Card%20%20Fraud%20Detection/Credit_Card_Fraud_Detection_Report.pdf) |
| [Employee Retention / Attrition](Data-Science/Employee%20Retention%20Data) | Predicts employee attrition and surfaces the key drivers for HR intervention. | [📄 PDF](Data-Science/Employee%20Retention%20Data/Employee_Attrition_Report.pdf) |
| [Energy Consumption Prediction](Data-Science/Energy%20consumption%20Pediction) | Time-series forecasting of energy consumption. | [📄 PDF](Data-Science/Energy%20consumption%20Pediction/Energy_Consumption_Prediction_Report.pdf) |
| [House Prices — Advanced Regression](Data-Science/House%20Regression) | Kaggle Ames house prices (RMSLE) — stacked ensemble of 7 tuned models + SHAP. | [📄 PDF](Data-Science/House%20Regression/House_Prices_Report.pdf) |
| [House Price Prediction](Data-Science/House%20pricing) | End-to-end house-price regression with feature engineering and SHAP explainability. | [📄 PDF](Data-Science/House%20pricing/House_Price_Prediction_Report.pdf) |
| [Network Intrusion Detection](Data-Science/Network%20Intrusion%20Detection) | Intrusion-detection system classifying malicious vs. benign network traffic. | [📄 PDF](Data-Science/Network%20Intrusion%20Detection/IDS_Report.pdf) |
| [Online Retail Segmentation](Data-Science/On%20Line%20Retail%20Segmentation) | RFM + clustering customer segmentation with a product-recommendation layer. | [📄 PDF](Data-Science/On%20Line%20Retail%20Segmentation/Customer_Recommendation_Segmentation_Report.pdf) |
| [Stroke Risk Prediction](Data-Science/Stroke%20Disease%20Segmentation) | Clinical stroke-risk prediction on imbalanced health data. | [📄 PDF](Data-Science/Stroke%20Disease%20Segmentation/Stroke_Risk_Prediction_Report.pdf) |
| [Telecom Packages Segmentation](Data-Science/Telecom%20Packages%20Segmentation) | Telecom customer & package segmentation to inform targeted offers. | [📄 PDF](Data-Science/Telecom%20Packages%20Segmentation/Telecom_Segmentation_Report.pdf) |
| [Weather Forecast](Data-Science/Weather%20Forecast) | Weather forecasting pipeline with an interactive dashboard. | [📄 PDF](Data-Science/Weather%20Forecast/Weather_Forecast_Report.pdf) |

---

## 🗂️ Repository layout

```
AIML/                 deep learning · computer vision · NLP · LLM agents
Data-Science/         tabular ML · analytics · forecasting · segmentation
  └── <Project>/
        ├── <Project>_Report.pdf     ← final report (start here)
        ├── README.md                ← project-specific notes
        ├── requirements.txt
        └── src/ · notebooks/ · ...  ← code & pipeline
```

## 🛠️ Tech stack

Python · scikit-learn · PyTorch · XGBoost / LightGBM / CatBoost · pandas · matplotlib ·
SHAP · Grad-CAM · Streamlit · CrewAI · reportlab

## 👤 Author

**Triaz Malik** — [@triaz-malik](https://github.com/triaz-malik) · triaz.malik@gmail.com

## 📄 License

Released under the [MIT License](LICENSE).
