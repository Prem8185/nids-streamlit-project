import streamlit as st
import pandas as pd
import pickle

# Load saved files

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# Title

st.title("Network Intrusion Detection System")

st.write(
    "Upload a network traffic dataset and detect cyber attacks using Machine Learning."
)

# Upload CSV File

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    # Read Dataset

    dataset = pd.read_csv(uploaded_file)

    # Dataset Information

    st.subheader("Dataset Information")

    st.write("Total Records :", len(dataset))
    st.write("Total Columns :", len(dataset.columns))

    # Dataset Preview

    st.subheader("Dataset Preview (First 10 Rows)")

    st.dataframe(dataset.head(10))

    # Feature Selection

    x = dataset.iloc[:,0:-2].values

    # Encoding

    x = encoder.transform(x)

    # Scaling

    x = scaler.transform(x)

    # Prediction

    predictions = model.predict(x)

    attack_names = label_encoder.inverse_transform(predictions)

    # Add Predictions to Dataset

    result = dataset.copy()

    result["Predicted Attack"] = attack_names

    # Prediction Results

    st.subheader("Prediction Results")

    st.write(
        "Total Predictions Generated :",
        len(result)
    )

    st.write(
        "Showing First 50 Prediction Records"
    )

    st.dataframe(result.head(50))

    # Attack Distribution Graph

    st.subheader("Attack Distribution")

    attack_counts = (
        result["Predicted Attack"]
        .value_counts()
    )

    st.bar_chart(attack_counts)

    # Attack Summary

    st.subheader("Attack Summary")

    summary = attack_counts.reset_index()

    summary.columns = [
        "Attack Type",
        "Count"
    ]

    st.table(summary)

    # Download Results

    st.subheader("Download Results")

    csv = result.to_csv(index=False)

    st.download_button(
        label="Download Prediction CSV",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv"
    )
