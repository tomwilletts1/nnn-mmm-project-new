import streamlit as st
import pandas as pd
import numpy as np
import os
from src.embeddings_utils import embed_dataframe_column, embed_texts
from src.model import train_model
from frontend.ui_helpers import section_header

st.set_page_config(page_title="Marketing ML Demo", layout="wide")
st.title("📊 Marketing Data Machine Learning Demo")
st.markdown("""
This app demonstrates how marketing data can be processed, embedded, and used in machine learning models to predict sales and analyze creative impact.
""")

# --- Utility Functions ---
@st.cache_data(show_spinner=True)
def load_processed_data():
    path = "data/processed/processed_weekly_data.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        st.error(f"Processed data not found at {path}")
        return pd.DataFrame()

@st.cache_data(show_spinner=True)
def load_embeddings():
    path = "data/embeddings/creative_text_embeddings.csv"
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    else:
        return None

# --- Sidebar: Data Selection ---
st.sidebar.header("1️⃣ Data Selection")
data_file = st.sidebar.file_uploader("Upload your processed CSV (optional)", type=["csv"])
if data_file:
    df = pd.read_csv(data_file)
    st.success("Custom data loaded!")
else:
    df = load_processed_data()

if df.empty:
    st.stop()

section_header("Data Preview", icon="🔍")
st.dataframe(df.head(20), use_container_width=True)

# --- Embedding Generation ---
st.sidebar.header("2️⃣ Embedding Generation")
if st.sidebar.button("Generate Embeddings for Creative Text", help="Uses OpenAI API and may take time/cost credits."):
    try:
        with st.spinner("Generating embeddings for creative text..."):
            emb_df = embed_dataframe_column(df, "creative_text", "data/embeddings/creative_text_embeddings.csv")
        st.success("Embeddings generated and saved!")
    except Exception as e:
        st.error(f"Embedding generation failed: {e}\nCheck your OpenAI API key in .env or environment variables.")
        emb_df = None
else:
    emb_df = load_embeddings()
    if emb_df is not None:
        st.sidebar.success("Embeddings loaded from file.")
    else:
        st.sidebar.info("No embeddings found. Generate them above.")

# --- Merge Embeddings ---
if emb_df is not None:
    embedding_cols = [col for col in emb_df.columns if isinstance(col, int) or (isinstance(col, str) and col.isdigit())]
    df = df.merge(emb_df, left_on="creative_text", right_index=True, how="left")
else:
    embedding_cols = []

# --- Feature Selection ---
st.sidebar.header("3️⃣ Feature Selection")
default_features = ["spend", "impressions", "creative_length", "creative_word_count"]
all_features = default_features + [str(col) for col in embedding_cols]
feature_cols = st.sidebar.multiselect(
    "Select features for modeling",
    options=all_features,
    default=default_features + [str(col) for col in embedding_cols[:8]],
    help="You can select both structured and embedding features."
)

# --- Model Training ---
st.sidebar.header("4️⃣ Model Training & Evaluation")
if st.sidebar.button("Train Model"):
    if not feature_cols:
        st.error("Please select at least one feature.")
    elif "sales" not in df.columns:
        st.error("Target column 'sales' not found in data.")
    else:
        X = df[feature_cols].fillna(0).values
        y = df["sales"].values
        with st.spinner("Training model..."):
            model, preds = train_model(X, y)
        st.success("Model trained!")
        section_header("Model Evaluation", icon="📈")
        st.write(f"**Features used:** {feature_cols}")
        st.write(f"**Rows used:** {len(y)}")
        st.write("**Actual vs Predicted Sales (Test Set):**")
        chart_df = pd.DataFrame({"Actual": y[-len(preds):], "Predicted": preds})
        st.line_chart(chart_df)
        st.write(chart_df.describe())
else:
    st.info("Select features and click 'Train Model' to see results.")

# --- Predict for New Creative Text ---
st.sidebar.header("5️⃣ Predict New Creative")
new_text = st.sidebar.text_area("Enter new creative text for embedding and prediction:")
if new_text and emb_df is not None and feature_cols:
    try:
        with st.spinner("Embedding new creative text..."):
            new_emb = embed_texts([new_text])[0]
        # Build input for model (use mean of numeric features as placeholder)
        input_dict = {f: float(df[f].mean()) if f in df.columns else 0.0 for f in feature_cols}
        for i, col in enumerate([str(c) for c in embedding_cols]):
            if col in feature_cols:
                input_dict[col] = new_emb[i]
        input_vec = np.array([input_dict[f] for f in feature_cols]).reshape(1, -1)
        if 'model' in locals():
            pred = model.predict(input_vec)[0]
            st.sidebar.success(f"Predicted sales for new creative: {pred:.2f}")
        else:
            st.sidebar.info("Train a model first to enable prediction.")
    except Exception as e:
        st.sidebar.error(f"Embedding or prediction failed: {e}\nCheck your OpenAI API key.")
elif new_text:
    st.sidebar.info("Generate embeddings and train a model to enable prediction.")

# --- Footer ---
section_header("", icon="")
st.markdown("""
---
*Built with ❤️ using Streamlit, OpenAI, and scikit-learn.*
""") 