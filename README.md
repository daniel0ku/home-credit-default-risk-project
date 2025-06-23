# Home Credit Default Risk 🎯

Many people struggle to access credit due to limited or no credit history, often leading to financial exclusion or risky loans. **Home Credit Group** aims to solve this by providing safe, responsible lending using alternative data—like telecom and transaction records—to better evaluate repayment ability.

### 📖 Project Overview

Check out the `POS Home Credit Default Risk.pdf` file in this repository for a detailed guide on how the analysis was conducted and instructions on using the prediction API. It covers the full process—from data exploration to model deployment—and provides practical steps to test the API yourself.

## 🧪 Hypotheses

To guide feature creation and modeling, we tested these ideas:

* **No Kids, Less Risk**: People with no children (CNT_CHILDREN = 0) might have lower expenses and be less likely to default.
* **Past Delays, Higher Risk**: Applicants with high 30-day and 60-day delinquency ratios (DEF_30_RATIO and DEF_60_RATIO) could signal financial trouble and a greater chance of default.
* **Short Activity Bursts**: Applicants with activity mostly in short periods (e.g., hour/day) might default more than those with steady, long-term patterns (e.g., quarter/year).

### 📊 Hypotheses Results

* **No Kids:** Applicants without children showed better financial stability, likely due to fewer household costs.
* **Delinquency Ratios**: Those with high DEF_30_RATIO and DEF_60_RATIO had an 11.84% default rate, compared to 8.37% for others.
* **Activity Patterns**: Short-term activity (hour/day) was linked to higher default risk (p < 0.001), but differences in other timeframes (e.g., week/month) weren’t significant (p > 0.05).

🔍 *Note*: While these findings were meaningful, the features didn’t boost model performance much:

* CNT_CHILDREN was dropped (likely overlapped with CNT_FAM_MEMBERS).
* DEF_30_RATIO was skipped (possibly redundant with similar features).
* Hour/day ratio didn’t make the final cut, suggesting it added little unique value.
💡 Next Steps: Try new features or interactions for better predictions.

💡 *Exploring additional features, patterns, or interactions may yield better predictive value in future iterations.*

## ⚙️ Machine Learning Methods and Results

### Methods

We built a model to predict loan repayment risk using these steps:

1. **Data Prep:** Cleaned the data by filling missing values (e.g., medians for numbers, most frequent for categories) and reduced redundant features using an initial LightGBM model to spot low-value columns.
2. **Feature Engineering:** Added features from extra datasets (e.g., bureau.csv) like loan counts and averages, assuming higher IDs meant newer records since timestamps were missing.
3. **Model Choice:** Focused on LightGBM for its speed and ability to handle imbalanced data. Also tested XGBoost and CatBoost for comparison.
4. **Tuning:** Used Optuna to fine-tune LightGBM settings (e.g., tree depth, learning rate) for better performance, targeting AUC-ROC.
5. **Evaluation:** Checked results with stratified cross-validation, focusing on AUC-ROC, F1-score, and Recall to catch risky clients (TARGET = 1) despite the 8.1% vs. 91.9% class imbalance.
6. **Deployment:** Wrapped the final model in a FastAPI web app, making it easy to use with JSON inputs.

### Results

- **Performance:** The tuned LightGBM model nailed the default rate on the test set, matching training results closely (AUC consistent, F1 slightly lower).
- **Key Finding:** It’s great at spotting risky clients but flags more false positives—useful for caution, less so for precision.
- **Feature Impact:** Custom features (e.g., credit ratios) didn’t help as much as expected and were dropped to keep the model simple and effective.
- **API Ready:** The model is live at [this link](https://home-credit-optuna-model-982383075824.us-central1.run.app/docs)—send JSON data and get predictions, even with missing features.

## Evaluation Metric

Model performance is evaluated using the **Area Under the Receiver Operating Characteristic Curve (AUC-ROC)**. This metric reflects how well the model can distinguish between clients who will repay and those who will default.

## 💡 Suggestions and Recommendations

Here are some ideas to improve the project and make it even more effective:

1. **Better Feature Engineering:**
   - **Time-Based Features:** Since timestamps are missing, try creating more proxies for time (e.g., gaps between loan applications in bureau.csv) to capture trends or recent behavior.
   - **Interaction Features:** Test combinations like income vs. loan amount or delinquency vs. credit inquiries to uncover hidden patterns that single features might miss.
   - **External Data:** If possible, add economic indicators (e.g., unemployment rates) to reflect broader conditions affecting repayment.

2. **Handle Imbalance Smarter:**
   - **Oversampling/Undersampling:** Experiment with techniques like SMOTE or random undersampling to balance the 8.1% vs. 91.9% split—might boost Recall for risky clients.
   - **Custom Loss Function:** Tweak LightGBM’s loss function to penalize missing TARGET = 1 more, pushing the model to focus on the minority class.

3. **Model Upgrades:**
   - **Stacking:** Combine LightGBM, XGBoost, and CatBoost predictions with a simple meta-model (e.g., logistic regression) to see if blending improves results.
   - **Neural Networks:** Try a small neural net for comparison—could catch non-linear patterns that tree-based models miss, though it’ll need scaled data.

4. **API and Usability:**
   - **Prediction Explanations:** Add a feature to the API that explains why a prediction was made (e.g., top contributing factors), making it more useful for decision-makers.
   - **Batch Processing:** Let users send multiple records at once to the API for faster bulk predictions—great for real-world testing.

## 🗃️ Downloading the Dataset
Due to file size limitations on GitHub, the dataset is not included in this repository. Please follow these steps to download and set up the data:

1. Download the dataset from [Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data).
2. Unzip the downloaded archive.
3. Move all the .csv files into the data/ folder in the root of this repository.

Your directory structure should look like this:

<pre>
home-credit-default-risk/
├── .venv/
├── .vscode/
├── app/
├── catboost_info/
├── data/
├── data_summaries/
├── submissions/
├── eda/
├── custom_transformers.py
├── Dockerfile
├── eda_utilities.py
├── home_credit_default_risk_modeling.ipynb
├── lgbm_optuna.joblib
├── lgbm_optuna.py
├── POS Home Credit Default Risk.pdf
├── README.MD
└── requirements.txt
</pre>

## Installing Dependencies

Before running the project, make sure to install the required Python packages. All dependencies and their versions are listed in the requirements.txt file.

<pre>
pip install -r requirements.txt
</pre>

This will ensure your environment has all the necessary libraries to run the project smoothly.

## 👤 Author Information

**Name:** Daniel Kurmel

**Email:** danielkurmel@gmail.com

**Dataset Source:** [Kaggle – Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)