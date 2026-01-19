import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==================================================
# 🚀 Page Configuration
# ==================================================
st.set_page_config(
    page_title="🚗 Cars EDA Project",
    layout="wide"
)

# ==================================================
# 📦 Data Loading
# ==================================================
@st.cache_data
def load_raw():
    return pd.read_csv("Cars.csv")

@st.cache_data
def load_cleaned():
    df = load_raw().copy()
    df.drop_duplicates(inplace=True)
    for col in ["Price", "Power"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

raw = load_raw()
clean = load_cleaned()

"Welcome to te Car Data EDA Dashboard. This project is designed to perform complete Explatory Data Analysis(EDA) on a car dataset using python + Streamlit"
st.title("🚗 Cars Analytics Dashboard")
st.subheader("📊 Dataset Overview")
st.write("Columns in cleaned dataset:")
st.write(clean.columns)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Cars 🚘", len(clean))
c2.metric("Avg Price 💰", round(clean["Price"].mean(), 2) if "Price" in clean else "N/A")
c3.metric("Avg Power ⚡", round(clean["Power"].mean(), 2) if "Power" in clean else "N/A")
c4.metric("Features 🔧", clean.shape[1])

st.success("Dataset loaded successfully 🎉")

"Car  Data Analysis"
"* Data types"
"* Missing values"
"* Duplicate values"
"* Summary statistics"
"* Correlation matrix"
st.header("📊 Exploratory Data Analysis")
df = clean.copy()

num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(include="object").columns.tolist()

# --------------------------------------------------
# 🔢 Metrics
# --------------------------------------------------
st.subheader("Metrics")
k1, k2, k3 = st.columns(3)
k1.metric("Selected Cars 🚗", len(df))
k2.metric("Average Price 💰", round(df["Price"].mean(), 2) if "Price" in df else "N/A")
k3.metric("Average Power ⚡", round(df["Power"].mean(), 2) if "Power" in df else "N/A")

"EDA Dashboard"
"* Histogram"
"* Boxplot"
"* Bar chart"
"* Scatter plot"
"* Interactive fltering from sidebar"
st.subheader("📈 Univariate Analysis")
col = st.selectbox("Choose Column 🔽", df.columns)

fig, ax = plt.subplots()
if col in num_cols:
    sns.histplot(df[col], kde=True, ax=ax)
else:
    df[col].value_counts().plot(kind="bar", ax=ax)

st.pyplot(fig)

# --------------------------------------------------
# 📉 Bivariate Analysis
# --------------------------------------------------
st.subheader("📉 Bivariate Analysis")
x = st.selectbox("X Axis 📌", df.columns, key="bivar_x")
y = st.selectbox("Y Axis 📌", df.columns, key="bivar_y")

fig2, ax2 = plt.subplots()
if x in num_cols and y in num_cols:
    sns.scatterplot(data=df, x=x, y=y, ax=ax2)
else:
    sns.boxplot(data=df, x=x, y=y, ax=ax2)
st.pyplot(fig2)

# --------------------------------------------------
# 🔥 Multivariate Analysis
# --------------------------------------------------
st.subheader("🔥 Multivariate Analysis")
option = st.selectbox(
    "Choose Analysis Type 🎯",
    ["Heatmap", "Pairplot"]
)

if option == "Heatmap":
    if len(num_cols) > 1:
        fig3, ax3 = plt.subplots(figsize=(9, 5))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax3)
        st.pyplot(fig3)
    else:
        st.info("Not enough numeric columns for heatmap ⚠️")

elif option == "Pairplot":
    if len(num_cols) > 1:
        st.pyplot(sns.pairplot(df[num_cols]))
    else:
        st.info("Not enough numeric columns for pairplot ⚠️")

# ==================================================
# 📌 Conclusions Section
# ==================================================
st.header("📌 Automated Insights")
st.write("""
### 🔍 Key Findings:
- 💰 Price varies significantly across brands
- ⚡ Higher power often correlates with higher price
- 🚘 Fuel type impacts cost distribution
""")
st.success("EDA Completed Successfully 🎉🚀")
"TIP"
"Use the sidebar filters to explore a specific category and perform analysis on filtered data."