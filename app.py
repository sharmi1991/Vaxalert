"""
VaxAlert - Streamlit Web App
AI Agent for Vaccination Coverage & Awareness
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import os
import urllib.request
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="VaxAlert - AI Vaccination Agent",
    page_icon="💉",
    layout="wide"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0a1428; color: white; }
    .stApp { background-color: #0a1428; }
    h1 { color: #00e6aa !important; }
    h2 { color: #00c896 !important; }
    h3 { color: #00c896 !important; }
    .metric-box {
        background-color: #0a2030;
        border: 1px solid #00c896;
        border-radius: 8px;
        padding: 15px;
        margin: 5px;
    }
    .urgent { color: #ff4444; font-weight: bold; }
    .good { color: #00e6aa; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.title("💉 VaxAlert")
st.subheader("AI Agent for Vaccination Coverage & Awareness")
st.markdown("*Kaggle 5-Day AI Agents Intensive Capstone | Agents for Good*")
st.divider()

# ─────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/vaccinations.csv"
    urllib.request.urlretrieve(url, "vaccinations.csv")
    df = pd.read_csv("vaccinations.csv", parse_dates=["date"])
    return df

# ─────────────────────────────────────────────
# Tool 1: Data Analysis
# ─────────────────────────────────────────────
def identify_low_coverage(df, top_n=10):
    exclude = ["World", "Europe", "Asia", "Africa", "North America",
               "South America", "Oceania", "European Union",
               "High income", "Low income", "Upper middle income",
               "Lower middle income", "International"]
    latest = (df[["location", "date", "people_vaccinated_per_hundred"]]
              .fillna(0)
              .sort_values("date")
              .groupby("location").last()
              .reset_index())
    latest = latest[~latest["location"].isin(exclude)]
    latest = latest[latest["people_vaccinated_per_hundred"] > 0]
    latest = latest.sort_values("people_vaccinated_per_hundred")
    global_avg = round(latest["people_vaccinated_per_hundred"].mean(), 2)
    return latest.head(top_n), global_avg, latest

# ─────────────────────────────────────────────
# Tool 2: LightGBM Prediction
# ─────────────────────────────────────────────
def forecast_trend(df, country, forecast_days=30):
    df_c = df[df["location"] == country][["date", "people_vaccinated_per_hundred"]].copy()
    df_c = df_c.sort_values("date").dropna()

    if len(df_c) < 30:
        return None

    df_c["day_of_year"] = df_c["date"].dt.dayofyear
    df_c["month"] = df_c["date"].dt.month
    df_c["week"] = df_c["date"].dt.isocalendar().week.astype(int)
    for lag in range(1, 8):
        df_c[f"lag_{lag}"] = df_c["people_vaccinated_per_hundred"].shift(lag)
    df_c["rolling_7d"] = df_c["people_vaccinated_per_hundred"].shift(1).rolling(7).mean()
    df_c["rolling_14d"] = df_c["people_vaccinated_per_hundred"].shift(1).rolling(14).mean()
    df_c = df_c.dropna()

    features = ["day_of_year", "month", "week",
                "lag_1","lag_2","lag_3","lag_4","lag_5","lag_6","lag_7",
                "rolling_7d","rolling_14d"]

    X, y = df_c[features], df_c["people_vaccinated_per_hundred"]
    split = int(len(X) * 0.8)
    model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05,
                               num_leaves=31, verbose=-1)
    model.fit(X.iloc[:split], y.iloc[:split])

    history = df_c[["date","people_vaccinated_per_hundred"]].copy()
    preds = []
    last_date = history["date"].max()

    for i in range(1, forecast_days+1):
        next_date = last_date + pd.Timedelta(days=i)
        temp = history.copy()
        temp["day_of_year"] = temp["date"].dt.dayofyear
        temp["month"] = temp["date"].dt.month
        temp["week"] = temp["date"].dt.isocalendar().week.astype(int)
        for lag in range(1, 8):
            temp[f"lag_{lag}"] = temp["people_vaccinated_per_hundred"].shift(lag)
        temp["rolling_7d"] = temp["people_vaccinated_per_hundred"].shift(1).rolling(7).mean()
        temp["rolling_14d"] = temp["people_vaccinated_per_hundred"].shift(1).rolling(14).mean()
        temp = temp.dropna()
        if temp.empty:
            break
        pred = float(model.predict(temp[features].iloc[[-1]])[0])
        pred = min(max(pred, 0), 100)
        preds.append(pred)
        new_row = pd.DataFrame({"date":[next_date],
                                "people_vaccinated_per_hundred":[pred]})
        history = pd.concat([history, new_row], ignore_index=True)

    current = float(df_c["people_vaccinated_per_hundred"].iloc[-1])
    forecasted = round(preds[-1], 2) if preds else current
    change = forecasted - current
    trend = "improving" if change > 1 else "declining" if change < -1 else "stagnating"

    return {"current": round(current, 2), "forecasted": forecasted,
            "trend": trend, "change": round(change, 2),
            "history": df_c, "forecast_values": preds}

# ─────────────────────────────────────────────
# Tool 3: Campaign Generator
# ─────────────────────────────────────────────
def generate_campaign(country, coverage, trend, change):
    urgency = "HIGH" if coverage < 30 or trend == "declining" else "MEDIUM"
    
    # Try Gemini API
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"""
Country: {country}, Coverage: {coverage}%, Trend: {trend}, Urgency: {urgency}
Generate:
AWARENESS MESSAGE: (2-3 sentences, empathetic)
CAMPAIGN ACTION PLAN: (3 specific bullet points)
KEY METRIC: (one measurable indicator)
"""
            response = model.generate_content(prompt)
            return response.text, urgency
        except:
            pass

    # Demo mode
    return f"""
**AWARENESS MESSAGE:**
In {country}, vaccination coverage stands at only {coverage}%.
Every vaccine protects not just one person, but an entire community.
Together, we can build a healthier future for all.

**CAMPAIGN ACTION PLAN:**
1. Deploy mobile vaccination units to rural and remote areas
2. Partner with local community leaders for trust-based outreach
3. Launch SMS campaigns in local languages to address hesitancy

**KEY METRIC:**
Weekly vaccination rate — Target: +0.5% per week
""", urgency

# ─────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────
with st.spinner("Loading vaccination data..."):
    try:
        df = load_data()
        bottom_countries, global_avg, all_latest = identify_low_coverage(df)
        data_loaded = True
    except:
        st.error("Could not load data. Check internet connection.")
        data_loaded = False

if data_loaded:
    # Sidebar
    st.sidebar.title("🔍 VaxAlert Controls")
    st.sidebar.metric("Global Average Coverage", f"{global_avg}%")
    st.sidebar.divider()

    mode = st.sidebar.radio(
        "Select Mode:",
        ["🌍 Global Overview", "🔎 Country Analysis"]
    )

    # ── GLOBAL OVERVIEW ──
    if mode == "🌍 Global Overview":
        st.header("🌍 Global Vaccination Overview")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Global Average", f"{global_avg}%")
        with col2:
            st.metric("Countries Analyzed", len(all_latest))
        with col3:
            urgent = len(all_latest[all_latest["people_vaccinated_per_hundred"] < 30])
            st.metric("Urgent Countries (<30%)", urgent)

        st.subheader("Bottom 10 Countries — Lowest Vaccination Coverage")

        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.RdYlGn(np.linspace(0.1, 0.5, len(bottom_countries)))
        ax.barh(bottom_countries["location"],
                bottom_countries["people_vaccinated_per_hundred"],
                color=colors, edgecolor="white")
        ax.axvline(x=global_avg, color="cyan", linestyle="--",
                   linewidth=2, label=f"Global Avg: {global_avg}%")
        ax.set_xlabel("People Vaccinated Per Hundred (%)", color="white")
        ax.set_title("VaxAlert: Countries Needing Urgent Campaigns",
                     color="#00e6aa", fontsize=13, fontweight="bold")
        ax.set_facecolor("#0a1428")
        fig.patch.set_facecolor("#0a1428")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#00c896")
        ax.spines["left"].set_color("#00c896")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(facecolor="#0a2030", labelcolor="white")
        st.pyplot(fig)

        st.subheader("Bottom 10 Countries Data")
        st.dataframe(bottom_countries[["location", "people_vaccinated_per_hundred", "date"]]
                     .rename(columns={"location": "Country",
                                      "people_vaccinated_per_hundred": "Coverage (%)",
                                      "date": "Last Updated"}),
                     use_container_width=True)

    # ── COUNTRY ANALYSIS ──
    else:
        st.header("🔎 Country-Level Analysis")

        countries = sorted(all_latest["location"].tolist())
        selected = st.selectbox("Select Country:", countries,
                                index=countries.index("Burundi") if "Burundi" in countries else 0)

        if st.button("🚀 Run VaxAlert Agent", type="primary"):

            # Tool 1
            with st.spinner("Tool 1: Analyzing coverage data..."):
                country_data = all_latest[all_latest["location"] == selected]
                coverage = float(country_data["people_vaccinated_per_hundred"].iloc[0])

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Current Coverage", f"{coverage}%",
                          delta=f"{coverage - global_avg:.1f}% vs global avg")
            with col2:
                urgency_label = "🔴 URGENT" if coverage < 30 else "🟡 MEDIUM" if coverage < 60 else "🟢 GOOD"
                st.metric("Status", urgency_label)

            # Tool 2
            with st.spinner("Tool 2: Running LightGBM forecast..."):
                result = forecast_trend(df, selected)

            if result:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Coverage", f"{result['current']}%")
                with col2:
                    st.metric("30-day Forecast", f"{result['forecasted']}%",
                              delta=f"{result['change']:+.2f}%")
                with col3:
                    trend_emoji = {"improving": "📈", "declining": "📉", "stagnating": "➡️"}
                    st.metric("Trend", f"{trend_emoji[result['trend']]} {result['trend'].upper()}")

                # Forecast chart
                fig, ax = plt.subplots(figsize=(12, 4))
                hist = result["history"].tail(60)
                forecast_dates = [hist["date"].max() + pd.Timedelta(days=i+1)
                                  for i in range(len(result["forecast_values"]))]
                ax.plot(hist["date"], hist["people_vaccinated_per_hundred"],
                        color="#00c896", linewidth=2, label="Historical")
                ax.plot(forecast_dates, result["forecast_values"],
                        color="#ff9944", linewidth=2, linestyle="--", label="30-day Forecast")
                ax.axvline(x=hist["date"].max(), color="white", linestyle=":", alpha=0.5)
                ax.set_title(f"VaxAlert Prediction: {selected} — {result['trend'].upper()}",
                             color="#00e6aa", fontsize=12, fontweight="bold")
                ax.set_facecolor("#0a1428")
                fig.patch.set_facecolor("#0a1428")
                ax.tick_params(colors="white")
                ax.legend(facecolor="#0a2030", labelcolor="white")
                ax.spines["bottom"].set_color("#00c896")
                ax.spines["left"].set_color("#00c896")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                st.pyplot(fig)

            # Tool 3
            with st.spinner("Tool 3: Generating awareness campaign..."):
                campaign, urgency = generate_campaign(
                    selected, coverage,
                    result["trend"] if result else "stagnating",
                    result["change"] if result else 0
                )

            st.subheader(f"📢 Campaign Plan — Urgency: {'🔴 HIGH' if urgency == 'HIGH' else '🟡 MEDIUM'}")
            st.markdown(campaign)

            # Final summary
            st.divider()
            st.success(f"✅ VaxAlert Agent Complete! Data → Insight → Action for {selected}")
