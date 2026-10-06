
import io
import zipfile
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="AQI Safety Checker",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Accessible, high-contrast UI
# -----------------------------
st.markdown(
    """
    <style>
    /* ---------- Color palette ----------
       Deep navy  : #16324F
       Teal       : #177E89
       Mint       : #DFF3F1
       Cream      : #F8F7F2
       Dark text  : #17202A
       Focus      : #FFB703
    ------------------------------------ */

    :root {
        --navy: #16324F;
        --teal: #177E89;
        --mint: #DFF3F1;
        --cream: #F8F7F2;
        --text: #17202A;
        --focus: #FFB703;
        --white: #FFFFFF;
    }

    .stApp {
        background: var(--cream);
        color: var(--text);
    }

    /* Keep content comfortable to read */
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Headings */
    h1, h2, h3 {
        color: var(--navy) !important;
        letter-spacing: 0.2px;
    }

    p, li, label, .stMarkdown {
        color: var(--text);
        font-size: 1.02rem;
        line-height: 1.6;
    }

    /* Keyboard focus */
    button:focus-visible,
    input:focus-visible,
    textarea:focus-visible,
    [role="button"]:focus-visible,
    [role="combobox"]:focus-visible {
        outline: 4px solid var(--focus) !important;
        outline-offset: 3px !important;
    }

    /* Buttons */
    .stButton > button {
        min-height: 52px;
        border-radius: 12px;
        border: 2px solid var(--teal);
        background: var(--teal);
        color: var(--white);
        font-size: 1.05rem;
        font-weight: 700;
        padding: 0.65rem 1.2rem;
    }

    .stButton > button:hover {
        background: var(--navy);
        border-color: var(--navy);
        color: var(--white);
    }

    /* Inputs */
    input {
        min-height: 48px !important;
        font-size: 1.05rem !important;
        border-radius: 10px !important;
    }

    /* Cards */
    .hero {
        background: linear-gradient(135deg, var(--navy), var(--teal));
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 1.5rem;
    }

    .hero h1, .hero p {
        color: white !important;
    }

    .card {
        background: white;
        border: 1px solid #D8E3E7;
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 3px 12px rgba(22, 50, 79, 0.06);
    }

    .result-card {
        background: var(--mint);
        border-left: 7px solid var(--teal);
        border-radius: 14px;
        padding: 1.25rem;
        margin-top: 1rem;
    }

    .result-title {
        color: var(--navy);
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }

    .result-value {
        color: var(--navy);
        font-size: 2rem;
        font-weight: 800;
    }

    .help-box {
        background: #FFF8E7;
        border-left: 6px solid var(--focus);
        border-radius: 12px;
        padding: 1rem 1.15rem;
        color: var(--text);
    }

    /* Reduce animation for people sensitive to motion */
    @media (prefers-reduced-motion: reduce) {
        * {
            scroll-behavior: auto !important;
            transition: none !important;
            animation: none !important;
        }
    }

    /* Mobile */
    @media (max-width: 768px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .hero {
            padding: 1.25rem;
        }

        .result-value {
            font-size: 1.6rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Helper functions
# -----------------------------
@st.cache_data
def load_uploaded_file(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    """Load a CSV or a ZIP containing a CSV."""
    name = file_name.lower()

    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes))

    if name.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
            csv_files = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not csv_files:
                raise ValueError("The ZIP file does not contain a CSV file.")
            # Use the first CSV, matching the original notebook's single-dataset setup.
            with z.open(csv_files[0]) as f:
                return pd.read_csv(f)

    raise ValueError("Please upload a CSV file or a ZIP file containing a CSV.")


@st.cache_resource
def train_model(df: pd.DataFrame):
    """Train the same model used in the original notebook."""
    required = {"AQI Value", "Status"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "The uploaded dataset is missing: " + ", ".join(sorted(missing))
        )

    data = df[["AQI Value", "Status"]].copy()
    data["AQI Value"] = pd.to_numeric(data["AQI Value"], errors="coerce")
    data = data.dropna()

    if data.empty:
        raise ValueError("No valid AQI values were found in the dataset.")

    if data["Status"].nunique() < 2:
        raise ValueError("At least two Status categories are required for training.")

    X = data[["AQI Value"]]
    y = data["Status"]

    # Stratification is safer when every class has enough samples.
    stratify = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    report = classification_report(
        y_test,
        y_pred,
        labels=model.classes_,
        output_dict=True,
        zero_division=0,
    )

    return model, data, accuracy, cm, report


def status_message(status: str) -> str:
    """Give a simple explanation without relying on color alone."""
    value = str(status).strip().lower()

    if any(word in value for word in ["good", "safe", "satisfactory"]):
        return "This category is generally associated with better air quality."
    if any(word in value for word in ["moderate", "acceptable"]):
        return "Air quality is in a middle range. Sensitive people may want to be cautious."
    if any(word in value for word in ["poor", "unhealthy", "bad"]):
        return "Air quality may be unhealthy. Consider reducing prolonged outdoor exposure."
    if any(word in value for word in ["severe", "hazardous", "very unhealthy"]):
        return "This indicates a high-risk air-quality category. Take extra precautions."

    return "The model predicted this category from the AQI value in the uploaded dataset."


# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <a href="#aqi-checker" style="
        position:absolute; left:-10000px; top:auto; width:1px; height:1px;
        overflow:hidden;
    " onfocus="this.style.left='10px'; this.style.width='auto'; this.style.height='auto';">
        Skip to AQI Checker
    </a>

    <div class="hero" id="aqi-checker">
        <h1>🌿 AQI Safety Checker</h1>
        <p>
            Enter an Air Quality Index (AQI) value and get the predicted air-quality
            category using the Decision Tree model from your notebook.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Sidebar: simple navigation
# -----------------------------
with st.sidebar:
    st.markdown("## Navigation")
    page = st.radio(
        "Choose a section",
        ["AQI Checker", "Dataset & Model", "Model Performance", "Accessibility"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        """
        <div class="help-box">
        <strong>Keyboard tip:</strong><br>
        Use <b>Tab</b> to move between controls and <b>Enter/Space</b> to activate buttons.
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Dataset upload
# -----------------------------
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

if page == "AQI Checker":
    st.header("1. Load your dataset")

    st.markdown(
        """
        <div class="card">
        Upload the same dataset used in your notebook. The file must contain:
        <ul>
            <li><b>AQI Value</b> — numerical AQI value</li>
            <li><b>Status</b> — the category the model should predict</li>
        </ul>
        You can upload a <b>.csv</b> file or the <b>.zip</b> file containing the CSV.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload CSV or ZIP dataset",
        type=["csv", "zip"],
        help="Your file is processed by this Streamlit session. Required columns: AQI Value and Status.",
    )

    if uploaded is not None:
        try:
            df = load_uploaded_file(uploaded.getvalue(), uploaded.name)
            model, clean_df, accuracy, cm, report = train_model(df)

            st.session_state.data_loaded = True
            st.session_state.model = model
            st.session_state.clean_df = clean_df
            st.session_state.accuracy = accuracy
            st.session_state.cm = cm
            st.session_state.report = report

            st.success(
                f"Dataset loaded successfully: {len(clean_df):,} usable rows."
            )

        except Exception as exc:
            st.error(f"Could not load the dataset: {exc}")

    # -----------------------------
    # Prediction
    # -----------------------------
    st.header("2. Check an AQI value")

    if not st.session_state.data_loaded:
        st.info("Upload your dataset above to activate the AQI checker.")
    else:
        col1, col2 = st.columns([2, 1])

        with col1:
            aqi = st.number_input(
                "Enter AQI Value",
                min_value=0.0,
                max_value=1000.0,
                value=50.0,
                step=1.0,
                help="Enter the AQI number you want the model to classify.",
            )

        with col2:
            st.markdown(
                """
                <div class="card">
                <b>Input format</b><br>
                Whole numbers or decimals are accepted.<br><br>
                Example: <b>75</b>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if st.button("🔎 Check AQI", use_container_width=True):
            prediction = st.session_state.model.predict([[aqi]])[0]
            explanation = status_message(prediction)

            st.markdown(
                f"""
                <div class="result-card" role="status" aria-live="polite">
                    <div class="result-title">Predicted AQI Category</div>
                    <div class="result-value">{prediction}</div>
                    <p><b>AQI value:</b> {aqi:g}</p>
                    <p>{explanation}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="help-box">
            <strong>Important:</strong> This is a machine-learning prediction based on
            the uploaded training data. It should not be treated as a medical diagnosis.
            </div>
            """,
            unsafe_allow_html=True,
        )

elif page == "Dataset & Model":
    st.header("Dataset & Model")

    if not st.session_state.data_loaded:
        st.warning("Go to AQI Checker and upload your dataset first.")
    else:
        df = st.session_state.clean_df
        model = st.session_state.model

        c1, c2, c3 = st.columns(3)
        c1.metric("Usable rows", f"{len(df):,}")
        c2.metric("AQI feature", "AQI Value")
        c3.metric("Categories", f"{df['Status'].nunique()}")

        st.subheader("Dataset preview")
        st.dataframe(df.head(20), use_container_width=True, hide_index=True)

        st.subheader("Category counts")
        counts = df["Status"].value_counts().rename_axis("Status").reset_index(name="Count")
        st.dataframe(counts, use_container_width=True, hide_index=True)

        st.subheader("Model used")
        st.code(
            """DecisionTreeClassifier(random_state=42)

Feature:
    AQI Value

Target:
    Status""",
            language="text",
        )

elif page == "Model Performance":
    st.header("Model Performance")

    if not st.session_state.data_loaded:
        st.warning("Go to AQI Checker and upload your dataset first.")
    else:
        accuracy = st.session_state.accuracy
        cm = st.session_state.cm
        model = st.session_state.model
        report = st.session_state.report

        st.metric("Test Accuracy", f"{accuracy * 100:.2f}%")

        st.subheader("Confusion Matrix")
        cm_df = pd.DataFrame(
            cm,
            index=[f"Actual: {c}" for c in model.classes_],
            columns=[f"Predicted: {c}" for c in model.classes_],
        )
        st.dataframe(cm_df, use_container_width=True)

        st.subheader("Classification Report")
        report_df = pd.DataFrame(report).transpose().round(3)
        st.dataframe(report_df, use_container_width=True)

        st.caption(
            "Accuracy and classification metrics are calculated on the held-out test set "
            "using random_state=42, matching the original notebook setup."
        )

elif page == "Accessibility":
    st.header("Accessibility")

    st.markdown(
        """
        <div class="card">
        <h3>Designed for easier access</h3>
        <ul>
            <li><b>Keyboard-friendly:</b> controls can be reached with the Tab key.</li>
            <li><b>Large controls:</b> buttons and input fields are intentionally larger for easier selection.</li>
            <li><b>High contrast:</b> text and controls use strong contrast instead of relying on color alone.</li>
            <li><b>Clear labels:</b> important inputs have visible labels and help text.</li>
            <li><b>Screen-reader friendly:</b> important prediction output is announced as a status update where supported.</li>
            <li><b>Reduced motion:</b> the interface respects the user's reduced-motion preference.</li>
            <li><b>Simple navigation:</b> the application is divided into four clear sections.</li>
            <li><b>Mobile-friendly:</b> the layout adapts to smaller screens.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Recommended assistive-technology use")
    st.markdown(
        """
        For users with visual or motor disabilities, this interface works best with:
        - keyboard navigation,
        - browser zoom,
        - a screen reader such as NVDA or Narrator,
        - and the operating system's accessibility settings.
        """
    )

    st.markdown(
        """
        <div class="help-box">
        <b>Design principle:</b> no important result is communicated through color alone.
        The predicted category is always written as text.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")
st.caption("AQI Safety Checker • Decision Tree model • Accessible Streamlit interface")
