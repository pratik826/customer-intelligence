import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# =========================
# SESSION HISTORY INIT
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# LOAD MODEL & DATA
# =========================
model = joblib.load("model/model.pkl")

data = pd.read_csv("data/raw_data.csv")
data['Date'] = pd.to_datetime(data['Date'])

USD_TO_INR = 83

# global averages (INR)
avg_spent_global = data['Total_Cost'].mean() * USD_TO_INR
avg_items_global = data['Total_Items'].mean()

# handle NaN
if pd.isna(avg_items_global):
    avg_items_global = 0

# =========================
# TITLE
# =========================
st.title("🛒 Customer Intelligence Dashboard")
st.write("Analyze customer behavior and predict high-value potential")

# =========================
# SIDEBAR INPUT
# =========================
st.sidebar.header("Enter Customer Details")

mode = st.sidebar.radio("Select Mode", ["Choose Existing Customer", "Manual Input"])

if mode == "Choose Existing Customer":
    customer_list = data['Customer_Name'].unique()
    customer_name = st.sidebar.selectbox("Select Customer", customer_list)
else:
    customer_name = st.sidebar.text_input("Customer Name", "New Customer")

# =========================
# INPUT FEATURES (INR UI)
# =========================
if mode == "Manual Input":
    avg_spent_inr = st.sidebar.slider("Avg Spend (₹ per transaction)", 0, 20000, 4000)
    avg_items = st.sidebar.slider("Avg Items per Transaction", 1.0, 10.0, 5.0)
    purchase_count = st.sidebar.slider("Total Purchase Count", 1, 20, 3)
    recency = st.sidebar.slider("Recency (days since last purchase)", 0, 1000, 200)
    discount_usage = st.sidebar.slider("Discount Usage (%)", 0.0, 1.0, 0.5)

# =========================
# AUTO-FILL (convert to INR)
# =========================
if mode == "Choose Existing Customer":
    cust_data = data[data['Customer_Name'] == customer_name]

    if not cust_data.empty:
        avg_spent = cust_data['Total_Cost'].mean()
        avg_spent_inr = avg_spent * USD_TO_INR

        avg_items = cust_data['Total_Items'].mean()
        if pd.isna(avg_items):
            avg_items = 0

        purchase_count = len(cust_data)

        last_date = cust_data['Date'].max()
        recency = (data['Date'].max() - last_date).days

        discount_usage = cust_data['Discount_Applied'].mean()

# =========================
# CONVERT INR → USD (MODEL INPUT)
# =========================
avg_spent = avg_spent_inr / USD_TO_INR

input_data = np.array([[avg_spent, avg_items, purchase_count, recency, discount_usage]])

# =========================
# HISTORY SIDEBAR
# =========================
st.sidebar.subheader("📜 Analysis History")

if len(st.session_state.history) == 0:
    st.sidebar.write("No customers analyzed yet.")
else:
    for record in reversed(st.session_state.history[-5:]):
        st.sidebar.write(
            f"{record['name']} → {record['prediction']} ({record['probability']:.2f})"
        )

# =========================
# HELPER FUNCTIONS
# =========================
def get_frequency_label(purchase_count):
    if purchase_count > 8:
        return "High"
    elif purchase_count > 3:
        return "Medium"
    else:
        return "Low"

def get_recency_label(recency):
    if recency < 150:
        return "Recent"
    elif recency < 400:
        return "Moderate"
    else:
        return "Inactive"

def get_segment(purchase_count, recency):
    if purchase_count > 8 and recency < 200:
        return "High-Value Segment"
    elif purchase_count > 2:
        return "Mid-Value Segment"
    else:
        return "Low-Value Segment"

def get_recommendation(pred):
    if pred == 1:
        return [
            "Offer loyalty rewards",
            "Provide early access to premium products",
            "Avoid unnecessary discounts"
        ]
    else:
        return [
            "Run reactivation campaigns",
            "Offer targeted discounts",
            "Send personalized reminders"
        ]

# =========================
# MAIN BUTTON
# =========================
if st.button("Analyze Customer"):

    prob = model.predict_proba(input_data)[0][1]
    pred = model.predict(input_data)[0]

    prob_display = min(max(prob, 0.01), 0.99)

    # SAVE HISTORY
    st.session_state.history.append({
        "name": customer_name,
        "prediction": "High Value" if pred == 1 else "Low Value",
        "probability": prob_display
    })

    st.subheader(f"👤 Customer Profile: {customer_name}")

    tab1, tab2 = st.tabs(["📊 Analysis", "📜 Customer History"])

    # =========================
    # TAB 1
    # =========================
    with tab1:

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Prediction", "High Value" if pred == 1 else "Low Value")

        with col2:
            st.metric("Probability", f"{prob_display:.2f}")
            st.progress(int(prob_display * 100))

        with col3:
            st.metric("Segment", get_segment(purchase_count, recency))

        st.subheader("📌 Behavioral Profile")

        st.write(f"- Purchase Frequency: **{get_frequency_label(purchase_count)} ({purchase_count} transactions)**")
        st.write(f"- Customer Activity: **{get_recency_label(recency)} ({recency} days)**")
        st.write(f"- Average Spending: **₹{avg_spent_inr:.0f} per transaction**")
        st.write(f"- Discount Usage: **{discount_usage*100:.0f}% of purchases**")

        # =========================
        # COLORED CHARTS
        # =========================
        st.subheader("📈 Comparison with Average Customer")

        col1, col2 = st.columns(2)

        # Spend Chart
        with col1:
            fig1, ax1 = plt.subplots()
            values = [avg_spent_inr, avg_spent_global]

            ax1.bar(['User', 'Average'], values, color=['#4CAF50', '#2196F3'])
            ax1.set_title("Avg Spend (₹)")

            for i, v in enumerate(values):
                ax1.text(i, v + max(values)*0.02, f"{v:.0f}", ha='center')

            st.pyplot(fig1)

        # Items Chart
        with col2:
            fig2, ax2 = plt.subplots()
            values = [avg_items, avg_items_global]

            values = [v if v > 0 else 0.1 for v in values]

            ax2.bar(['User', 'Average'], values, color=['#FF9800', '#9C27B0'])
            ax2.set_title("Avg Items per Transaction")

            ax2.set_ylim(0, max(values) * 1.5)

            for i, v in enumerate(values):
                ax2.text(i, v + 0.1, f"{v:.1f}", ha='center')

            st.pyplot(fig2)

        # =========================
        # EXPLANATION
        # =========================
        st.subheader("🧠 Model Explanation")

        if pred == 1:
            st.write("High purchase frequency and recent activity indicate strong future spending potential.")
        else:
            st.write("Low engagement and/or inactivity reduce future spending likelihood.")

        # =========================
        # RECOMMENDATIONS
        # =========================
        st.subheader("💡 Business Recommendations")

        for rec in get_recommendation(pred):
            st.write(f"- {rec}")

    # =========================
    # TAB 2
    # =========================
    with tab2:

        st.subheader("📜 Customer Transaction History")

        customer_history = data[data['Customer_Name'] == customer_name]

        if customer_history.empty:
            st.warning("No historical data found for this customer.")
        else:
            st.write("### Summary")
            st.write(f"- Total Transactions: {len(customer_history)}")
            st.write(f"- Total Spend: ₹{customer_history['Total_Cost'].sum()*USD_TO_INR:.0f}")
            st.write(f"- First Purchase: {customer_history['Date'].min()}")
            st.write(f"- Last Purchase: {customer_history['Date'].max()}")

            monthly = customer_history.groupby(
                customer_history['Date'].dt.to_period('M')
            ).sum(numeric_only=True)

            st.line_chart(monthly['Total_Cost'])

            st.dataframe(customer_history.head(20))