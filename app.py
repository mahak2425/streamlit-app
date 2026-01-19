import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Cars EDA Project", layout="wide")


@st.cache_data
def load_raw():
    return pd.read_csv("Cars.csv")

@st.cache_data
def load_cleaned():
    return pd.read_csv("Cars_cleaned.csv")

st.write("Columns in dataset:")
st.write(df.columns)

raw = load_raw()
clean = load_cleaned()

page = st.sidebar.radio("Navigation",
["Introduction","Analysis","Conclusions"])


if page == "Introduction":

    st.title("Cars Analytics Dashboard")

    c1,c2,c3,c4 = st.columns(4)

    c1.metric("Total Cars", len(clean))
    c2.metric("Average Price", round(clean["Price"].mean(),2))
    c3.metric("Average KM", int(clean["Kilometers_Driven"].mean()))
    c4.metric("Total Companies", clean["Company_Name"].nunique())


    st.subheader("Raw Dataset")
    st.dataframe(raw, use_container_width=True)

    st.subheader("Cleaned Dataset")
    st.dataframe(clean, use_container_width=True)


    st.subheader("Location Map")

    if "Latitude" in clean.columns and "Longitude" in clean.columns:
        st.map(clean[["Latitude","Longitude"]])
    else:
        st.info("Latitude and Longitude not available")




elif page == "Analysis":

    st.title("Exploratory Analysis Studio")

    company = st.sidebar.multiselect(
        "Select Company",
        options=clean["Company_Name"].unique(),
        default=clean["Company_Name"].unique()
    )

    year = st.sidebar.slider(
        "Select Year Range",
        int(clean["Year"].min()),
        int(clean["Year"].max()),
        (int(clean["Year"].min()), int(clean["Year"].max()))
    )

    df = clean[(clean["Company_Name"].isin(company)) &
               (clean["Year"].between(year[0],year[1]))]


    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()


    k1,k2,k3 = st.columns(3)

    k1.metric("Selected Cars", len(df))
    k2.metric("Average Price", round(df["Power"].mean(),2))
    k3.metric("Average Power", round(df["Power_Value"].mean(),2))

    st.header("Univariate Analysis")

    col = st.selectbox("Choose Column", df.columns)

    fig, ax = plt.subplots(figsize=(7,4))

    if col in cat_cols:
        sns.countplot(y=df[col], ax=ax)

    else:
        dist = st.radio("View Type",["Histogram","KDE","Boxplot"])

        if dist=="Histogram":
            sns.histplot(df[col], kde=True, ax=ax)

        elif dist=="KDE":
            sns.kdeplot(df[col], fill=True, ax=ax)

        else:
            sns.boxplot(x=df[col], ax=ax)

    st.pyplot(fig)



    st.header("Bivariate Analysis")

    c1,c2 = st.columns(2)

    x = c1.selectbox("X Axis", df.columns)
    y = c2.selectbox("Y Axis", df.columns)

    fig2,ax2 = plt.subplots(figsize=(7,4))

    if x in num_cols and y in num_cols:
        sns.scatterplot(data=df,x=x,y=y,ax=ax2)
        st.write("Correlation:", round(df[x].corr(df[y]),3))

    elif x in num_cols and y in cat_cols:
        sns.boxplot(data=df,x=y,y=x,ax=ax2)

    elif x in cat_cols and y in num_cols:
        sns.boxplot(data=df,x=x,y=y,ax=ax2)

    else:
        sns.countplot(data=df,x=x,hue=y,ax=ax2)

    st.pyplot(fig2)



    st.header("Multivariate Analysis")

    option = st.selectbox("Method",
    ["Heatmap","Pairplot","Grouped Bar"])


    if option=="Heatmap":
        fig3,ax3 = plt.subplots(figsize=(9,5))
        sns.heatmap(df[num_cols].corr(),annot=True,cmap="coolwarm",ax=ax3)
        st.pyplot(fig3)

    elif option=="Pairplot":
        pair = sns.pairplot(df[num_cols])
        st.pyplot(pair)

    else:
        if "Fuel_Type" in df.columns and "Price" in df.columns:

            fig4,ax4 = plt.subplots(figsize=(8,4))

            sns.barplot(
                data=df,
                x="Fuel_Type",
                y="Price",
                hue="Transmission" if "Transmission" in df.columns else None,
                ax=ax4
            )

            st.pyplot(fig4)

        else:
            st.warning("Required columns missing")




else:

    st.title("Automated Insights")

    st.write("Total Records:", len(clean))

    st.write("Highest Price Car:",
             clean.loc[clean["Price"].idxmax(),"Company_Name"])

    st.write("Most Common Fuel:",
             clean["Fuel_Type"].mode()[0])

    st.write("Strongest Correlation with Price:",
             clean.select_dtypes(include=np.number)
             .corr()["Price"].sort_values(ascending=False).index[1])
