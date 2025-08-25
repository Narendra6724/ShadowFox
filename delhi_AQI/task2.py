
import os
import io
import math
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# NLP (optional if you have a text column)
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter

warnings.filterwarnings("ignore")
plt.style.use("dark_background")
sns.set_theme(style="darkgrid")
sns.set_palette("bright")


st.set_page_config(page_title="Delhi AQI Explorer", layout="wide")
# Inject custom dark CSS for Streamlit UI
dark_css = """
<style>
/* Main app background */
.stApp {
    background-color: #121212;
    color: #E0E0E0;
}

/* Sidebar */
.css-1d391kg, .stSidebar {
    background-color: #1E1E1E !important;
    color: #E0E0E0 !important;
}

/* Metrics */
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
    color: #00E676 !important; /* neon green */
}

/* Headers and markdown text */
h1, h2, h3, h4, h5, h6, .stMarkdown {
    color: #FFFFFF !important;
}

/* Input widgets */
.css-1n543e5, .css-1x8cf1d, .stSelectbox, .stMultiSelect, .stTextInput {
    background-color: #2C2C2C !important;
    color: #E0E0E0 !important;
    border-radius: 6px;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-thumb {
    background: #555;
    border-radius: 10px;
}
</style>
"""
st.markdown(dark_css, unsafe_allow_html=True)
# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def load_csv(file_buffer) -> pd.DataFrame:
    df = pd.read_csv(file_buffer)
    return df

def parse_dates(df):
    # Try common date column names
    date_cols = [c for c in df.columns if c.lower() in ["date", "datetime", "timestamp", "measurementdate"]]
    if not date_cols:
        # attempt parse if any column looks like date
        for c in df.columns:
            try:
                parsed = pd.to_datetime(df[c], errors="raise", dayfirst=True, infer_datetime_format=True)
                df["Date"] = parsed
                break
            except Exception:
                continue
    else:
        df["Date"] = pd.to_datetime(df[date_cols[0]], errors="coerce", dayfirst=True, infer_datetime_format=True)
    if "Date" not in df:
        st.warning("Could not find/parse a date column. Some time-based charts will be disabled.")
        return df
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["MonthName"] = df["Date"].dt.month_name()
    df["Day"] = df["Date"].dt.day
    # Seasons adapted for North India / Delhi
    def season_from_month(m):
        if m in [12,1,2]:
            return "Winter (Dec-Feb)"
        elif m in [3,4,5]:
            return "Pre-Monsoon (Mar-May)"
        elif m in [6,7,8,9]:
            return "Monsoon (Jun-Sep)"
        else:
            return "Post-Monsoon (Oct-Nov)"
    if "Month" in df:
        df["Season"] = df["Month"].apply(season_from_month)
    return df

def infer_pollutants(df):
    # Common Indian AQI pollutants (CPCB/DPCC datasets)
    candidates = [
        "PM2.5","PM2_5","PM25","pm2_5","pm25","PM10","pm10",
        "NO2","no2",
        "SO2","so2",
        "CO","co",
        "O3","o3","Ozone","ozone",
        "NH3","nh3",
        "Benzene","Toluene","Xylene","benzen","toluene","xylene",
        "AQI","aqi","AQI_Bucket","aqi_bucket"
    ]
    cols = []
    for c in df.columns:
        if c in candidates or c.upper() in [x.upper() for x in candidates]:
            cols.append(c)
    # Add numeric-only columns likely to be pollutant concentrations
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    likely = []
    for c in num_cols:
        if c not in cols and (("pm" in c.lower()) or c.lower() in ["no2","so2","co","o3","nh3","aqi"]):
            likely.append(c)
    cols = cols + likely
    # De-duplicate while preserving order
    seen=set(); cols = [x for x in cols if not (x in seen or seen.add(x))]
    # Keep only numeric pollutant columns except AQI_Bucket
    pollutant_cols = []
    for c in cols:
        if c.lower() == "aqi_bucket" or c.lower() == "aqi category":
            continue
        if c in df and (pd.api.types.is_numeric_dtype(df[c]) or df[c].dtype==object):
            pollutant_cols.append(c)
    # Filter to numeric
    pollutant_cols = [c for c in pollutant_cols if c in df.columns and pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors="coerce"))]
    # Move AQI to front if present
    for aqi_name in ["AQI","aqi"]:
        if aqi_name in pollutant_cols:
            pollutant_cols.remove(aqi_name)
            pollutant_cols = [aqi_name] + pollutant_cols
            break
    return pollutant_cols

def ensure_nltk():
    try:
        _ = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")

def text_column_candidates(df):
    # Guess text columns that could contain comments/remarks
    patterns = ["remarks","comments","comment","notes","feedback","text","review","description"]
    txt = [c for c in df.columns if any(p in c.lower() for p in patterns)]
    # Also include non-numeric object columns (short)
    for c in df.columns:
        if df[c].dtype == object and c not in txt:
            # heuristic: average length > 8 suggests meaningful text
            try:
                avg_len = df[c].astype(str).str.len().mean()
                if avg_len >= 8:
                    txt.append(c)
            except Exception:
                pass
    return txt

def who_thresholds():
    # WHO AQG (2021) daily limits (approx; used for illustrative contribution pie)
    # Units assumed ug/m3 for PM, NO2, SO2; mg/m3 for CO; O3 8-hr mean ug/m3
    return {
        "PM2.5": 15.0,
        "PM25": 15.0,
        "PM2_5": 15.0,
        "PM10": 45.0,
        "NO2": 25.0,
        "SO2": 40.0,
        "O3": 100.0,
        "CO": 4.0,  # mg/m3 (if your data is in mg/m3)
        "NH3": 20.0
    }

def normalize_colname(c):
    return c.replace("_","").replace(".","").upper()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Delhi AQI Dashboard")
st.sidebar.markdown("Upload your CSV and choose filters. All charts use Matplotlib/Seaborn.")

uploaded = st.sidebar.file_uploader("Upload Delhi AQI CSV", type=["csv"])

default_path = "delhiaqi.csv"
df = None

if uploaded is not None:
    df = load_csv(uploaded)
elif os.path.exists(default_path):
    df = load_csv(default_path)
else:
    st.info("Upload a CSV to get started. Expected to have a date column and pollutant columns like PM2.5, PM10, NO2, SO2, CO, O3, NH3, etc.")
    st.stop()

# Parse dates and enrich
df = parse_dates(df)

# Infer pollutants
pollutant_cols = infer_pollutants(df)
if not pollutant_cols:
    st.error("Couldn't infer pollutant columns. Please ensure your CSV has numeric columns like PM2.5, PM10, NO2, SO2, CO, O3, NH3, AQI.")
    st.stop()

# Allow user to pick station/location if present
station_cols = [c for c in df.columns if any(k in c.lower() for k in ["station","location","site","area"])]
selected_station = None
if station_cols:
    selected_station = st.sidebar.selectbox("Station/Location (optional)", ["All"] + sorted(df[station_cols[0]].dropna().astype(str).unique().tolist()))
    if selected_station != "All":
        df = df[df[station_cols[0]].astype(str) == selected_station]

# Pollutant and month filters
default_selected = [x for x in pollutant_cols[:5]] if len(pollutant_cols)>=3 else pollutant_cols
selected_pollutants = st.sidebar.multiselect("Select pollutants", pollutant_cols, default=default_selected)

month_opts = []
if "MonthName" in df.columns:
    month_opts = df["MonthName"].dropna().unique().tolist()
    # Keep calendar order
    month_order = ["January","February","March","April","May","June","July","August","September","October","November","December"]
    month_opts = [m for m in month_order if m in month_opts]
selected_months = st.sidebar.multiselect("Select months", month_opts, default=month_opts if month_opts else [])

if selected_months and "MonthName" in df.columns:
    df = df[df["MonthName"].isin(selected_months)]

# -----------------------------
# Header / KPIs
# -----------------------------
st.title("Delhi Air Quality Index (AQI) Explorer")
st.caption("Analyze pollutants, seasonal variation, and geographic factors for Delhi.")

kpi_cols = st.columns(2)  # only 2 KPIs now
with kpi_cols[0]:
    st.metric("Rows", f"{len(df):,}")
with kpi_cols[1]:
    st.metric("Date Range", f"{str(df['Date'].min().date()) if 'Date' in df else 'N/A'} → {str(df['Date'].max().date()) if 'Date' in df else 'N/A'}")

st.markdown("---")

# -----------------------------
# Charts
# -----------------------------
left, right = st.columns([1, 2.2])

with left:
    st.subheader("Filters")
    st.write("Use the sidebar to upload data and filter months/pollutants.")
    if station_cols:
        st.write(f"Station column detected: **{station_cols[0]}**")

    st.subheader("NLP (Optional)")
    text_cols = text_column_candidates(df)
    choice = None
    if text_cols:
        choice = st.selectbox("Text column for NLP", ["(skip NLP)"] + text_cols)
    else:
        st.info("No obvious text column found (e.g., comments/remarks).")

    if choice and choice != "(skip NLP)":
        ensure_nltk()
        sia = SentimentIntensityAnalyzer()
        text_series = df[choice].dropna().astype(str)
        # Basic cleanup
        tokens = []
        for t in text_series:
            tokens += [w.lower() for w in nltk.word_tokenize(t)]
        stop = set(stopwords.words("english"))
        tokens = [w for w in tokens if w.isalpha() and w not in stop and len(w)>2]
        counts = Counter(tokens).most_common(25)
        words, freqs = zip(*counts) if counts else ([], [])

        st.write("Top keywords")
        fig_kw, ax_kw = plt.subplots()
        ax_kw.barh(words[::-1], freqs[::-1])
        ax_kw.set_xlabel("Frequency")
        ax_kw.set_ylabel("Word")
        ax_kw.set_title("Top Keywords")
        st.pyplot(fig_kw)

        # Sentiment
        sentiments = text_series.apply(lambda x: sia.polarity_scores(x)["compound"])
        fig_s, ax_s = plt.subplots()
        sns.histplot(sentiments, bins=30, kde=True, ax=ax_s)
        ax_s.set_title("Sentiment (VADER compound)")
        ax_s.set_xlabel("Sentiment score [-1, 1]")
        st.pyplot(fig_s)
    else:
        st.caption("NLP: Select a text column to see keywords and sentiment.")

with right:
    st.subheader("Visualizations")

    # Time series line chart
    if "Date" in df.columns and selected_pollutants:
        fig1, ax1 = plt.subplots()
        for p in selected_pollutants:
            if p in df.columns:
                ax1.plot(df["Date"], pd.to_numeric(df[p], errors="coerce").rolling(window=7, min_periods=1).mean(), label=p)
        ax1.set_title("7-day Rolling Average over Time")
        ax1.set_xlabel("Date"); ax1.set_ylabel("Concentration")
        ax1.legend(loc="upper left", fontsize="small")
        st.pyplot(fig1)
    else:
        st.info("Add a Date column and choose pollutants to see time series.")

    # Monthly bar chart (avg per month)
    if "MonthName" in df.columns and selected_pollutants:
        monthly = df.groupby("MonthName")[selected_pollutants].mean(numeric_only=True)
        # Ensure calendar order
        month_order = ["January","February","March","April","May","June","July","August","September","October","November","December"]
        monthly = monthly.reindex([m for m in month_order if m in monthly.index])
        fig2, ax2 = plt.subplots()
        monthly.plot(kind="bar", ax=ax2)
        ax2.set_title("Average Concentration by Month")
        ax2.set_xlabel("Month"); ax2.set_ylabel("Mean concentration")
        st.pyplot(fig2)

    # Histogram for distribution
    if selected_pollutants:
        fig3, ax3 = plt.subplots()
        for p in selected_pollutants[:5]:  # limit to 5 to avoid clutter
            sns.histplot(pd.to_numeric(df[p], errors="coerce"), bins=30, kde=True, ax=ax3, label=p, alpha=0.35)
        ax3.set_title("Distribution of Selected Pollutants")
        ax3.set_xlabel("Concentration")
        ax3.legend()
        st.pyplot(fig3)

    # Box plot by season
    if "Season" in df.columns and selected_pollutants:
        long_df = df.melt(id_vars=["Season"], value_vars=selected_pollutants, var_name="Pollutant", value_name="Value")
        long_df["Value"] = pd.to_numeric(long_df["Value"], errors="coerce")
        fig4, ax4 = plt.subplots()
        sns.boxplot(data=long_df, x="Season", y="Value", hue="Pollutant", ax=ax4)
        ax4.set_title("Seasonal Distribution")
        ax4.set_xlabel("Season"); ax4.set_ylabel("Concentration")
        ax4.legend(loc="upper right", fontsize="small")
        st.pyplot(fig4)

    # Heatmap of correlations
    if selected_pollutants:
        corr_df = df[selected_pollutants].apply(pd.to_numeric, errors="coerce").corr()
        fig5, ax5 = plt.subplots()
        sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", ax=ax5)
        ax5.set_title("Correlation Heatmap (Selected Pollutants)")
        st.pyplot(fig5)

    # Scatter plot (choose X vs Y)
    if selected_pollutants and len(selected_pollutants) >= 2:
        st.markdown("**Scatter Plot**")
        c1, c2 = st.columns(2)
        with c1:
            x_var = st.selectbox("X variable", selected_pollutants, index=0, key="xvar")
        with c2:
            y_var = st.selectbox("Y variable", selected_pollutants, index=1, key="yvar")
        fig6, ax6 = plt.subplots()
        ax6.scatter(pd.to_numeric(df[x_var], errors="coerce"), pd.to_numeric(df[y_var], errors="coerce"), alpha=0.5)
        ax6.set_xlabel(x_var); ax6.set_ylabel(y_var)
        ax6.set_title(f"{x_var} vs {y_var}")
        st.pyplot(fig6)

    # Pie chart: share of exceedance counts vs WHO AQG
    thresholds = who_thresholds()
    avail = [p for p in selected_pollutants if normalize_colname(p) in [normalize_colname(k) for k in thresholds.keys()]]
    if avail:
        exceed_counts = []
        labels = []
        for p in avail:
            # map threshold with flexible matching
            th = None
            for k,v in thresholds.items():
                if normalize_colname(k) == normalize_colname(p):
                    th = v; break
            if th is None: 
                continue
            vals = pd.to_numeric(df[p], errors="coerce")
            c = (vals > th).sum()
            exceed_counts.append(c)
            labels.append(p)
        if sum(exceed_counts) > 0:
            fig7, ax7 = plt.subplots()
            ax7.pie(exceed_counts, labels=labels, autopct="%1.0f%%", startangle=90)
            ax7.set_title("Share of Exceedance Days (vs WHO daily limits)")
            st.pyplot(fig7)

# -----------------------------
# Extra: Environmental drivers if present
# -----------------------------
st.markdown("---")
st.subheader("Environmental Drivers (if available)")
driver_cols = []
for cand in ["Temperature","Temp","RH","Humidity","WindSpeed","Wind_Speed","WindSpeed_kmph","WindDir","Wind_Direction","Pressure"]:
    if cand in df.columns:
        driver_cols.append(cand)

if driver_cols:
    cols = st.columns(len(driver_cols))
    for i, dc in enumerate(driver_cols):
        with cols[i]:
            if "AQI" in df.columns:
                figd, axd = plt.subplots()
                axd.scatter(pd.to_numeric(df[dc], errors="coerce"), pd.to_numeric(df["AQI"], errors="coerce"), alpha=0.5)
                axd.set_xlabel(dc); axd.set_ylabel("AQI")
                axd.set_title(f"AQI vs {dc}")
                st.pyplot(figd)
else:
    st.caption("No environmental driver columns like Temperature, Humidity, WindSpeed found.")

# -----------------------------
# Footnote
# -----------------------------
st.markdown("""
-> Use the left sidebar to upload a new CSV, choose months, and select pollutants.
->The NLP panel (left) becomes active when you pick a text column (e.g., remarks/comments).
-> Charts are built with Matplotlib/Seaborn; feel free to customize styles.

""")
