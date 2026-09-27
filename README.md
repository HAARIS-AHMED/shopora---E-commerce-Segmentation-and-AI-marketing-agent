Shopora Customer Analytics

Two connected tools built on the same e-commerce order dataset (ecommerce_customer_behavior_dataset_v2.csv): a churn prediction model and an RFM segmentation dashboard with AI-generated marketing copy.

1. Customer Churn Prediction (customerchurn.ipynb)

Predicts which customers are likely to churn, trained on order-level e-commerce data.

Approach

The raw dataset has one row per order, with no explicit churn label. A customer-level table is built by aggregating orders (spend, order count, session behavior, ratings, favorite category/payment/device, etc.).
Churn is defined as: a customer's most recent order is more than 90 days before the latest date in the dataset — a standard recency-based definition used when no ground-truth label exists.
Recency itself is excluded from the model features (it's only used to define the label) to avoid target leakage.
Model: a RandomForestClassifier (300 trees, balanced class weights) inside a full sklearn Pipeline with median imputation + scaling for numeric features and most-frequent imputation + one-hot encoding for categorical features.
Evaluated with accuracy, ROC AUC, confusion matrix, and classification report; feature importances are extracted to show what actually drives churn.
The trained pipeline is saved as churn_model.pkl for reuse.

Output: churn_model.pkl — a ready-to-load churn classifier, plus comparison.csv of actual vs. predicted labels on the test set.

2. RFM Segmentation & Marketing App (rfm_marketing_app.py)

A Streamlit dashboard that segments customers by purchase behavior and drafts marketing emails for each segment using an LLM.

Approach

Cleans and loads customer order data (uploaded CSV or bundled default).
Computes RFM metrics (Recency, Frequency, Monetary) per customer, plus a KMeans cluster (configurable 2–8 clusters).
Buckets customers into RFM quartile scores and maps them to named segments: Champions, Loyal Customers, Big Spenders, At Risk, Regular.
Visualizes segment distribution and a Recency-vs-Monetary scatter plot colored by segment; full RFM table is browsable in-app.
For a selected segment, calls the Groq API (openai/gpt-oss-120b) to generate a ready-to-send marketing email (subject line, fixed CTA, brand voice) tailored to that segment.

Requirements: streamlit, pandas, scikit-learn, seaborn, matplotlib, groq, python-dotenv, and a Groq API key (via .env as GROQ_API_KEY or entered in the sidebar).

Run it:

bash
streamlit run rfm_marketing_app.py
Relationship Between the Two

The notebook answers "will this customer churn?" with a supervised model; the app answers "what kind of customer is this, and how do I talk to them?" with unsupervised segmentation. Both start from the same raw order-level dataset and the same RFM-style customer aggregation logic, so they're natural companions: churn risk scores from the notebook could be layered onto the app's segments (e.g., to prioritize "At Risk" customers who are also flagged as likely to churn).
