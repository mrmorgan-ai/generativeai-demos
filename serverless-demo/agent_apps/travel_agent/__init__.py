import json
import logging
import os
import re
from datetime import datetime, timedelta
from textwrap import dedent
from typing import Optional

import azure.functions as func
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.run.agent import RunOutput
from agno.tools.serpapi import SerpApiTools
from dotenv import find_dotenv, load_dotenv
from icalendar import Calendar, Event


dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# Get API keys from environment variables
openai_api_key = os.environ.get("OPENAI_API_KEY")
serp_api_key = os.environ.get("SERPAPI_KEY")


def generate_ics_content(
    plan_text: str, start_date: Optional[datetime] = None
) -> bytes:
    """
    Generate an ICS calendar file from a travel itinerary text.

    Args:
        plan_text: The travel itinerary text
        start_date: Optional start date for the itinerary (defaults to today)

    Returns:
        bytes: The ICS file content as bytes
    """
    cal = Calendar()
    cal.add("prodid", "-//AI Travel Planner//github.com//")
    cal.add("version", "2.0")

    if start_date is None:
        start_date = datetime.today()

    # Split the plan into days
    day_pattern = re.compile(r"Day (\d+)[:\s]+(.*?)(?=Day \d+|$)", re.DOTALL)
    days = day_pattern.findall(plan_text)

    if not days:
        event = Event()
        event.add("summary", "Travel Itinerary")
        event.add("description", plan_text)
        event.add("dtstart", start_date.date())
        event.add("dtend", start_date.date())
        event.add("dtstamp", datetime.now())
        cal.add_component(event)
    else:
        for day_num, day_content in days:
            day_num = int(day_num)
            current_date = start_date + timedelta(days=day_num - 1)

            event = Event()
            event.add("summary", f"Day {day_num} Itinerary")
            event.add("description", day_content.strip())
            event.add("dtstart", current_date.date())
            event.add("dtend", current_date.date())
            event.add("dtstamp", datetime.now())
            cal.add_component(event)

    return cal.to_ical()


def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Azure Function main entry point for travel itinerary generation.

    Expected request body:
    {
        "destination": "Paris",
        "num_days": 7,
        "start_date": "2025-01-01" (optional)
    }
    """
    logging.info("Python HTTP trigger function processed a request.")

    try:
        # Parse request body
        req_body = req.get_json()
        destination = req_body.get("destination")
        num_days = req_body.get("num_days")
        start_date_str = req_body.get("start_date", "2025-11-26")

        # Validate required parameters
        if not destination or not num_days:
            return func.HttpResponse(
                json.dumps(
                    {
                        "error": "Missing required parameters. Please provide 'destination' and 'num_days'."
                    }
                ),
                status_code=400,
                mimetype="application/json",
            )

        # Parse start date if provided
        start_date = None
        if start_date_str:
            try:
                start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            except ValueError:
                return func.HttpResponse(
                    json.dumps({"error": "Invalid date format. Use YYYY-MM-DD."}),
                    status_code=400,
                    mimetype="application/json",
                )

        # Check if API keys are available
        if not openai_api_key or not serp_api_key:
            return func.HttpResponse(
                json.dumps(
                    {
                        "error": "API keys not configured. Please set OPENAI_API_KEY and SERPAPI_KEY."
                    }
                ),
                status_code=500,
                mimetype="application/json",
            )

        # Initialize agents
        researcher = Agent(
            name="Researcher",
            role="Searches for travel destinations, activities, and accommodations based on user preferences",
            model=OpenAIChat(id="gpt-5-nano", api_key=openai_api_key),
            description=dedent(
                """\
            You are a world-class travel researcher. Given a travel destination and the number of days the user wants to travel for,
            generate a list of search terms for finding relevant travel activities and accommodations.
            Then search the web for each term, analyze the results, and return the 10 most relevant results.
            """
            ),
            instructions=[
                "Given a travel destination and the number of days the user wants to travel for, first generate a list of 3 search terms related to that destination and the number of days.",
                "For each search term, `search_google` and analyze the results."
                "From the results of all searches, return the 10 most relevant results to the user's preferences.",
                "Remember: the quality of the results is important.",
            ],
            tools=[SerpApiTools(api_key=serp_api_key)],
            add_datetime_to_context=True,
        )

        planner = Agent(
            name="Planner",
            role="Generates a draft itinerary based on user preferences and research results",
            model=OpenAIChat(id="gpt-5-nano", api_key=openai_api_key),
            description=dedent(
                """\
            You are a senior travel planner. Given a travel destination, the number of days the user wants to travel for, and a list of research results,
            your goal is to generate a draft itinerary that meets the user's needs and preferences.
            """
            ),
            instructions=[
                "Given a travel destination, the number of days the user wants to travel for, and a list of research results, generate a draft itinerary that includes suggested activities and accommodations.",
                "Ensure the itinerary is well-structured, informative, and engaging.",
                "Ensure you provide a nuanced and balanced itinerary, quoting facts where possible.",
                "Remember: the quality of the itinerary is important.",
                "Focus on clarity, coherence, and overall quality.",
                "Never make up facts or plagiarize. Always provide proper attribution.",
            ],
            add_datetime_to_context=True,
        )

        logging.info(f"Researching destination: {destination} for {num_days} days")

        # Execute research
        research_results: RunOutput = researcher.run(
            f"Research {destination} for a {num_days} day trip", stream=False
        )

        logging.info("Research completed, generating itinerary")

        # Generate itinerary
        prompt = f"""
        Destination: {destination}
        Duration: {num_days} days
        Research Results: {research_results.content}
        
        Please create a detailed itinerary based on this research.
        """

        response: RunOutput = planner.run(prompt, stream=False)
        itinerary_text = response.content

        logging.info("Itinerary generated successfully")

        # Generate ICS content
        ics_content = generate_ics_content(itinerary_text, start_date) # type: ignore

        # Return response with both text and ICS file
        return func.HttpResponse(
            json.dumps(
                {
                    "itinerary": itinerary_text,
                    "ics_file": ics_content.decode("utf-8"),
                    "destination": destination,
                    "num_days": num_days,
                }
            ),
            status_code=200,
            mimetype="application/json",
        )

    except ValueError as e:
        logging.error(f"ValueError: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": f"Invalid request format: {str(e)}"}),
            status_code=400,
            mimetype="application/json",
        )
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": f"Internal server error: {str(e)}"}),
            status_code=500,
            mimetype="application/json",
        ) # type: ignore
