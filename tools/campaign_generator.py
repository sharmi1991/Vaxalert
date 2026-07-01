"""
VaxAlert - Tool 3: Awareness Campaign Generator
Uses Gemini API to generate culturally sensitive,
targeted vaccination awareness messages and campaign plans.
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


def setup_gemini():
    """Initialize Gemini API with key from environment."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found. "
            "Please set it in your .env file. Never hardcode API keys!"
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")


def generate_campaign(
    country: str,
    current_coverage: float,
    trend: str,
    trend_change_pct: float = 0.0,
    target_audience: str = "general public"
) -> dict:
    """
    VaxAlert Campaign Generator Tool.
    Generates a targeted vaccination awareness message and 3-point action plan
    using the Gemini API.

    Args:
        country: Country name
        current_coverage: Current vaccination rate (%)
        trend: "improving" | "declining" | "stagnating"
        trend_change_pct: Predicted change in coverage over 30 days
        target_audience: Who the message targets (default: "general public")

    Returns:
        dict with keys:
            - country: str
            - awareness_message: str (the generated public health message)
            - action_plan: str (3-point campaign action plan for health workers)
            - urgency_level: str ("HIGH" | "MEDIUM" | "LOW")
            - full_output: str
    """
    try:
        model = setup_gemini()

        # Determine urgency based on coverage + trend
        if current_coverage < 30 or trend == "declining":
            urgency = "HIGH"
        elif current_coverage < 60 or trend == "stagnating":
            urgency = "MEDIUM"
        else:
            urgency = "LOW"

        # Build prompt for Gemini
        prompt = f"""
You are a public health communication expert helping design vaccination awareness campaigns.

Country: {country}
Current vaccination coverage: {current_coverage}%
30-day forecast trend: {trend} ({trend_change_pct:+.1f}% change predicted)
Target audience: {target_audience}
Urgency level: {urgency}

Please generate:

1. AWARENESS MESSAGE (2-3 sentences):
A clear, empathetic, and culturally sensitive public health message encouraging vaccination.
Avoid fear-based language. Focus on community protection and individual benefit.
Tailor the tone to the urgency level.

2. CAMPAIGN ACTION PLAN (exactly 3 bullet points):
Practical steps that local health workers and NGOs can take in the next 30 days
to improve vaccination coverage in {country}. Be specific and actionable.

3. KEY METRIC TO TRACK:
One measurable indicator to evaluate campaign success.

Format your response clearly with the section headers:
AWARENESS MESSAGE:
CAMPAIGN ACTION PLAN:
KEY METRIC:
"""

        response = model.generate_content(prompt)
        full_output = response.text.strip()

        # Parse sections from Gemini response
        awareness_msg = ""
        action_plan = ""
        key_metric = ""

        if "AWARENESS MESSAGE:" in full_output:
            parts = full_output.split("AWARENESS MESSAGE:")
            rest = parts[1]
            if "CAMPAIGN ACTION PLAN:" in rest:
                awareness_msg = rest.split("CAMPAIGN ACTION PLAN:")[0].strip()
                rest2 = rest.split("CAMPAIGN ACTION PLAN:")[1]
                if "KEY METRIC:" in rest2:
                    action_plan = rest2.split("KEY METRIC:")[0].strip()
                    key_metric = rest2.split("KEY METRIC:")[1].strip()
                else:
                    action_plan = rest2.strip()

        return {
            "country": country,
            "urgency_level": urgency,
            "awareness_message": awareness_msg or full_output,
            "action_plan": action_plan,
            "key_metric": key_metric,
            "full_output": full_output
        }

    except ValueError as e:
        return {
            "error": str(e),
            "country": country,
            "urgency_level": "UNKNOWN",
            "awareness_message": "",
            "action_plan": "",
            "full_output": ""
        }
    except Exception as e:
        return {
            "error": f"Gemini API call failed: {str(e)}",
            "country": country,
            "urgency_level": "UNKNOWN",
            "awareness_message": "",
            "action_plan": "",
            "full_output": ""
        }
