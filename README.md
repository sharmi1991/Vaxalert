
# 💉 VaxAlert — AI Agent for Vaccination Awareness

> **Kaggle 5-Day AI Agents Intensive Capstone | Agents for Good Track**

VaxAlert is an AI-powered agent that identifies countries with low vaccination coverage, forecasts 30-day trends using LightGBM, and auto-generates targeted public health awareness campaigns using the Gemini API — all orchestrated by a Google ADK agent.

---

## 🎯 Problem Statement

Millions remain unvaccinated due to lack of awareness and poor outreach targeting. Public health workers struggle to identify which regions need urgent attention and what messaging would work best. Manual data analysis is slow, and traditional dashboards only show the problem — they don't take action.

**VaxAlert solves this by going from data → insight → action automatically.**

---

## 🏗️ Architecture

```
User Query (natural language)
         │
         ▼
  ┌─────────────────────┐
  │   VaxAlert Agent    │  ← Google ADK orchestrator
  │   (Gemini 1.5 Flash)│
  └──────────┬──────────┘
             │
     ┌───────┼───────┐
     ▼       ▼       ▼
  Tool 1  Tool 2  Tool 3
  Data    LGBM    Gemini
  Analysis Predict Campaign
     │       │       │
     └───────┴───────┘
             │
             ▼
   Structured Report +
   Campaign Action Plan
```

### Agent Tools

| Tool | Function | Technology |
|------|----------|------------|
| Data Analysis | Identify low-coverage countries | pandas |
| Prediction | 30-day trend forecast | LightGBM |
| Campaign Generator | Awareness messages + action plan | Gemini API |

---

## 🧠 Key Concepts Applied

| Concept | Where |
|---------|-------|
| ✅ Agent / Multi-agent (ADK) | `agent.py` — full ADK agent with Runner + Session |
| ✅ Agent Skills / Tool Use | 3 registered tools with typed arguments |
| ✅ Security Features | API keys via `.env`, never hardcoded; input sanitization |
| ✅ Deployability | `requirements.txt`, setup instructions, local + Cloud Run ready |

---

## 📦 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/vaxalert.git
cd vaxalert
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys

```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

**Never commit your `.env` file. It is in `.gitignore`.**

### 4. Download the Dataset

Download the COVID-19 World Vaccination Progress dataset from Kaggle:
- URL: https://www.kaggle.com/datasets/gpreda/covid-world-vaccination-progress
- Save as: `data/vaccinations.csv`

```bash
mkdir data
# Place vaccinations.csv inside data/
```

### 5. Run the Agent

```bash
python agent.py
```

Or import and use programmatically:

```python
from agent import run_vaxalert_agent

response = run_vaxalert_agent(
    "Which countries have the lowest vaccination coverage "
    "and what campaigns should we run?"
)
print(response)
```

---

## 📊 Dataset

- - **Source:** [COVID-19 World Vaccination Progress Data](https://www.kaggle.com/datasets/fedesoriano/covid-19-vaccination-progress)
- **Key columns used:** `location`, `date`, `people_vaccinated_per_hundred`, `daily_vaccinations`
---

## 🔐 Security

- All API keys stored in `.env` (never committed)
- `.env` is listed in `.gitignore`
- Input country names are validated against dataset before processing
- Gemini API calls wrapped in try/except to prevent key exposure in errors

---

## 📁 Project Structure

```
vaxalert/
├── agent.py                  # Main ADK agent (entry point)
├── tools/
│   ├── __init__.py
│   ├── data_analysis.py      # Tool 1: Low coverage identification
│   ├── prediction.py         # Tool 2: LightGBM trend forecast
│   └── campaign_generator.py # Tool 3: Gemini campaign generator
├── data/
│   └── vaccinations.csv      # Dataset (not committed to GitHub)
├── requirements.txt
├── .env.example              # Template for API keys
├── .gitignore
└── README.md
```

---

## 🚀 Value Delivered

| Stakeholder | Benefit |
|-------------|---------|
| Public Health Officials | Instant identification of at-risk regions |
| NGOs & Campaign Managers | Ready-to-use awareness content |
| Policymakers | ML-based trend forecasts for resource planning |

---

## 👩‍💻 Built By

Built as part of **Kaggle's 5-Day AI Agents Intensive Course with Google**.

Track: **Agents for Good**
