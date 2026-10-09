# 🛒 Customer Intelligence Dashboard

An interactive machine learning dashboard that analyzes customer purchasing behavior and predicts whether a customer belongs to the **High Value** or **Low Value** category.

The application helps businesses understand customer activity, compare spending patterns, identify customer segments, and generate actionable business recommendations.

## 🚀 Live Demo

**Live Application:** [Open Customer Intelligence Dashboard](https://customer-intelligence.streamlit.app/)

> If your actual Streamlit URL is different, replace the link above with your deployed app URL.

## ✨ Features

- **Customer Analysis:** Select an existing customer or enter customer details manually.
- **Customer Value Prediction:** Predict high-value and low-value customers using a trained machine learning model.
- **Probability Score:** Display the model's predicted probability.
- **Behavioral Analysis:** Analyze purchase frequency, recency, average spending, and discount usage.
- **Customer Segmentation:** Categorize customers into high-value, mid-value, and low-value segments.
- **Spending Comparison:** Compare customer spending and average items per transaction against overall customer averages.
- **Transaction History:** Review historical transactions, total spending, and purchase dates.
- **Business Recommendations:** Suggest customer engagement strategies based on the prediction.
- **Analysis History:** View recently analyzed customers during the current app session.

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Web Framework:** Streamlit
- **Machine Learning:** XGBoost and scikit-learn
- **Data Processing:** Pandas, NumPy
- **Model Serialization:** Joblib
- **Data Visualization:** Matplotlib and Streamlit charts
- **Version Control:** Git and GitHub
- **Deployment:** Streamlit Community Cloud
- **Large File Storage:** Git LFS

## 📂 Project Structure

```text
customer-intelligence/
├── app.py
├── requirements.txt
├── data/
│   └── raw_data.csv
├── model/
│   ├── model.pkl
│   └── scaler.pkl
└── script/
    └── datapreprocessing.py
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/pratik826/customer-intelligence.git
cd customer-intelligence
```

### 2. Install Git LFS

The project dataset is stored using Git Large File Storage (LFS).

Install Git LFS from [git-lfs.com](https://git-lfs.com/), then run:

```bash
git lfs install
git lfs pull
```

### 3. Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app.py
```

The dashboard will open in your browser, usually at `http://localhost:8501`.

## 📊 How It Works

1. Select an existing customer or choose manual input.
2. Provide or review customer purchasing information.
3. Click **Analyze Customer**.
4. View the predicted customer value category and probability.
5. Explore behavioral metrics, spending comparisons, customer history, and recommendations.

### Customer attributes

The application uses these input features:

- Average spending per transaction
- Average items per transaction
- Total purchase count
- Recency (days since the last purchase)
- Discount usage

The model receives the average spending value in USD, while the dashboard displays spending in Indian rupees (INR).

## 🎯 Business Use Cases

- Identify potentially high-value customers.
- Understand customer purchasing patterns.
- Support customer retention strategies.
- Plan targeted discounts and reactivation campaigns.
- Make data-informed customer engagement decisions.

## 🔮 Future Improvements

- Add model explainability using SHAP.
- Introduce interactive customer filtering and advanced analytics.
- Add downloadable customer analysis reports.
- Track prediction history persistently across sessions.
- Improve model monitoring and evaluation metrics.

## 👨‍💻 Author

**Pratik Redekar**

GitHub: [@pratik826](https://github.com/pratik826)

---

*Developed as a machine learning project focused on customer behavior analysis and customer value prediction.*
