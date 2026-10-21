from smolagents import CodeAgent, GoogleSearchTool, VisitWebpageTool, OpenAIServerModel
from smolagents.utils import encode_image_base64, make_image_url

from PIL import Image
import os
from dotenv import load_dotenv
from huggingface_hub import login
import math
from typing import Optional, Tuple

from smolagents import tool

login()
load_dotenv()

@tool
def calculate_cargo_travel_time(
    origin_coords: Tuple[float, float],
    destination_coords: Tuple[float, float],
    # Average speed for cargo planes
    cruising_speed_kmh: Optional[float] = 750.0,
) -> float:
    """
    Calculate the travel time for a cargo plane between two points on Earth using great-circle distance.

    Args:
        origin_coords: Tuple of (latitude, longitude) for the starting point
        destination_coords: Tuple of (latitude, longitude) for the destination
        cruising_speed_kmh: Optional cruising speed in km/h (defaults to 750 km/h for typical cargo planes)

    Returns:
        float: The estimated travel time in hours

    Example:
        >>> # Chicago (41.8781° N, 87.6298° W) to Sydney (33.8688° S, 151.2093° E)
        >>> result = calculate_cargo_travel_time((41.8781, -87.6298), (-33.8688, 151.2093))
    """

    def to_radians(degrees: float) -> float:
        return degrees * (math.pi / 180)

    # Extract coordinates
    lat1, lon1 = map(to_radians, origin_coords)
    lat2, lon2 = map(to_radians, destination_coords)

    # Earth's radius in kilometers
    EARTH_RADIUS_KM = 6371.0

    # Calculate great-circle distance using the haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    distance = EARTH_RADIUS_KM * c

    # Add 10% to account for non-direct routes and air traffic controls
    actual_distance = distance * 1.1

    # Calculate flight time
    # Add 1 hour for takeoff and landing procedures
    flight_time = (actual_distance / cruising_speed_kmh) + 1.0

    # Format the results
    return round(flight_time, 2)



api_key = os.environ.get("OPEN_ROUTER_KEY", "")
if api_key == "":
    raise ValueError("api key not found")

open_router_base_url = os.environ.get("OPEN_ROUTER_BASE_URL", "")
if open_router_base_url == "":
    raise ValueError("open router url is required!")

serper_api_key = os.environ.get("SERPER_API_KEY", "")
if serper_api_key == "":
    raise ValueError("serper api key is required!")

def check_reasoning_and_plot(final_answer, agent_memory):
    multimodal_model = OpenAIServerModel(
        model_id="deepseek/deepseek-v4.1-flash",
        api_key=api_key,
        api_base=open_router_base_url,   
    )
    filepath = "saved_map.png"
    assert os.path.exists(filepath), "Make sure to save the plot under saved_map.png!"
    image = Image.open(filepath)
    prompt = (
        f"Here is a user-given task and the agent steps: {agent_memory.get_succinct_steps()}. Now here is the plot that was made."
        "Please check that the reasoning process and plot are correct: do they correctly answer the given task?"
        "First list reasons why yes/no, then write your final decision: PASS in caps lock if it is satisfactory, FAIL if it is not."
        "Don't be harsh: if the plot mostly solves the task, it should pass."
        "To pass, a plot should be made using px.scatter_map and not any other method (scatter_map looks nicer)."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": make_image_url(encode_image_base64(image))},
                },
            ],
        }
    ]
    output = multimodal_model(messages).content
    print("Feedback: ", output)
    if "FAIL" in output:
        raise Exception(output)
    return True

web_agent_model = OpenAIServerModel(
    model_id="deepseek/deepseek-v4.1-flash",
    api_key=api_key,
    api_base=open_router_base_url,
)
web_agent = CodeAgent(
    model=web_agent_model,
    tools=[
        GoogleSearchTool(serper_api_key),
        VisitWebpageTool(),
        calculate_cargo_travel_time,
    ],
    name="web_agent",
    description="Browses the web to find information",
    verbosity_level=0,
    max_steps=10,
)

manager_agent_model = OpenAIServerModel(
    model_id="deepseek/deepseek-v4.1-flash-1017",
    api_key=api_key,
    api_base=open_router_base_url,
)
manager_agent = CodeAgent(
    model=manager_agent_model,
    name="manager_agent",
    managed_agents=[web_agent],
    tools=[calculate_cargo_travel_time],
    additional_authorized_imports=[
        "geopandas",
        "plotly",
        "shapely",
        "json",
        "pandas",
        "numpy",
    ],
    planning_interval=5,
    verbosity_level=2,
    final_answer_checks=[check_reasoning_and_plot],
    max_steps=15
)

task = """
Find 10 valid locations in total:
- Batman filming locations around the world.
- Factories or production centers of supercar manufacturers.

Gotham coordinates:
latitude = 40.7128
longitude = -74.0060

For each location:
1. Research its real name and geographic location.
2. Find its approximate latitude and longitude.
3. Identify its category.
4. Calculate estimated cargo flight time to Gotham using
   calculate_cargo_travel_time.
5. Pass the location coordinates as origin_coords and
   Gotham coordinates as destination_coords.

Create a pandas DataFrame with these columns:
- name
- location
- category
- latitude
- longitude
- cargo_flight_time_hours

Save the DataFrame to results.csv using index=False.

Create a world map using plotly.express.scatter_map.
Requirements:
- Latitude: latitude
- Longitude: longitude
- Hover label: name
- Marker color: cargo_flight_time_hours
- Use a continuous color scale.
- Include at least 6 valid locations.

Save the map as saved_map.png.
Use fig.write_image("saved_map.png").

Return a concise summary of the results.
Do not invent coordinates or travel times.
"""

result = manager_agent.run(task)

manager_agent.visualize()

print(result)

fig = manager_agent.python_executor.state["fig"]

fig.write_html("saved_map.html")
fig.write_image("saved_map.png")