"""
VaxAlert - Main Agent
Smart Vaccination Awareness Agent using Google ADK.

This agent orchestrates three tools to:
1. Identify low vaccination coverage countries (Data Analysis Tool)
2. Forecast 30-day vaccination trends (LightGBM Prediction Tool)
3. Generate targeted awareness campaigns (Gemini Campaign Generator Tool)

Usage:
    python agent.py
    or
    from agent import run_vaxalert_agent
    run_vaxalert_agent("Which countries need urgent vaccination campaigns?")
"""

import os
from dotenv import load_dotenv

# Load environment variables — NEVER hardcode API keys
load_dotenv()

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

from tools.data_analysis import identify_low_coverage_countries
from tools.prediction import forecast_next_30_days
from tools.campaign_generator import generate_campaign


# ─────────────────────────────────────────────
# Tool 1: Data Analysis
# ─────────────────────────────────────────────
def analyze_vaccination_data(top_n: int = 5) -> str:
    """
    Identify the countries with the lowest vaccination coverage.
    
    Args:
        top_n: Number of low-coverage countries to return (default: 5)
    
    Returns:
        A summary string of low-coverage countries and global average.
    """
    result = identify_low_coverage_countries(top_n=top_n)
    if "error" in result:
        return f"Error: {result['error']}"
    return result["analysis_summary"]


# ─────────────────────────────────────────────
# Tool 2: LightGBM Prediction
# ─────────────────────────────────────────────
def predict_vaccination_trend(country: str) -> str:
    """
    Forecast the 30-day vaccination coverage trend for a specific country.
    
    Args:
        country: The country name to forecast (e.g., "Nigeria", "Haiti")
    
    Returns:
        A summary string with current coverage, forecasted coverage, and trend direction.
    """
    result = forecast_next_30_days(country=country)
    if "error" in result:
        return f"Error: {result['error']}"
    return result["forecast_summary"]


# ─────────────────────────────────────────────
# Tool 3: Campaign Generator
# ─────────────────────────────────────────────
def create_awareness_campaign(
    country: str,
    current_coverage: float,
    trend: str
) -> str:
    """
    Generate a targeted vaccination awareness message and campaign action plan for a country.
    
    Args:
        country: Country name
        current_coverage: Current vaccination rate as a percentage (e.g., 12.5)
        trend: Trend direction — "improving", "declining", or "stagnating"
    
    Returns:
        Full campaign output including awareness message, action plan, and key metric.
    """
    result = generate_campaign(
        country=country,
        current_coverage=current_coverage,
        trend=trend
    )
    if "error" in result:
        return f"Error: {result['error']}"
    return result["full_output"]


# ─────────────────────────────────────────────
# Build the VaxAlert ADK Agent
# ─────────────────────────────────────────────
def create_vaxalert_agent() -> Agent:
    """
    Create and return the VaxAlert ADK Agent with all three tools registered.
    """
    agent = Agent(
        name="VaxAlert",
        model="gemini-1.5-flash",
        description=(
            "VaxAlert is an AI agent that analyzes global vaccination coverage data, "
            "predicts 30-day vaccination trends using machine learning, and generates "
            "targeted public health awareness campaigns for low-coverage countries."
        ),
        instruction=(
            "You are VaxAlert, a public health AI agent specializing in vaccination awareness.\n\n"
            "When a user asks about vaccination campaigns or low-coverage countries:\n"
            "1. FIRST use analyze_vaccination_data to identify the lowest-coverage countries.\n"
            "2. THEN use predict_vaccination_trend for the top priority country identified.\n"
            "3. FINALLY use create_awareness_campaign to generate actionable campaign content.\n\n"
            "Always present your findings in a clear, structured format:\n"
            "- Start with a brief problem statement\n"
            "- Show the data analysis results\n"
            "- Share the trend forecast\n"
            "- Present the campaign plan\n"
            "- End with a clear call to action\n\n"
            "Be empathetic, factual, and action-oriented. "
            "Your goal is to help public health workers take informed action quickly."
        ),
        tools=[
            analyze_vaccination_data,
            predict_vaccination_trend,
            create_awareness_campaign,
        ],
    )
    return agent


# ─────────────────────────────────────────────
# Runner — Entry Point
# ─────────────────────────────────────────────
def run_vaxalert_agent(user_query: str) -> str:
    """
    Run the VaxAlert agent with a user query and return the response.
    
    Args:
        user_query: Natural language question about vaccination campaigns
    
    Returns:
        Agent's full response as a string
    """
    agent = create_vaxalert_agent()
    session_service = InMemorySessionService()

    APP_NAME = "vaxalert"
    USER_ID = "health_worker_01"
    SESSION_ID = "session_001"

    # Create session
    session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )

    runner = Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=session_service
    )

    # Wrap user query in ADK Content format
    message = Content(role="user", parts=[Part(text=user_query)])

    print(f"\n{'='*60}")
    print(f"VaxAlert Agent")
    print(f"Query: {user_query}")
    print(f"{'='*60}\n")

    final_response = ""
    for event in runner.run(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=message
    ):
        # Print tool calls as they happen (transparency)
        if hasattr(event, "tool_call") and event.tool_call:
            print(f"[Tool Called] {event.tool_call.name}")
        
        # Capture final text response
        if event.is_final_response():
            for part in event.content.parts:
                if hasattr(part, "text"):
                    final_response += part.text

    print(f"\n{'='*60}")
    print("FINAL REPORT:")
    print(f"{'='*60}")
    print(final_response)
    return final_response


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    query = (
        "Which countries have the lowest vaccination coverage? "
        "What is the trend forecast, and what awareness campaigns should we run?"
    )
    run_vaxalert_agent(query)
