import os
from dotenv import load_dotenv

from pydantic import BaseModel
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


# -------------------------------
# Step 1: Define a tool function
# -------------------------------
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

# -------------------------------
# Step 2: Configure the LLM
# -------------------------------
model = init_chat_model(
    "anthropic:claude-3-7-sonnet-latest",
    temperature=0
)

# -------------------------------
# Step 3: Define structured output schema
# -------------------------------
class WeatherResponse(BaseModel):
    conditions: str

# -------------------------------
# Step 4: Enable memory with checkpointer
# -------------------------------
checkpointer = InMemorySaver()

# -------------------------------
# Step 5: Create the agent
# -------------------------------
agent = create_react_agent(
    model=model,
    tools=[get_weather],
    prompt="You are a helpful assistant.",
    checkpointer=checkpointer,
    response_format=WeatherResponse
)

# -------------------------------
# Step 6: Run the agent with multi-turn conversation
# -------------------------------
# Unique conversation/session ID
config = {"configurable": {"thread_id": "1"}}

# First question
sf_response = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in San Francisco?"}]},
    config
)
print("SF Response:", sf_response["structured_response"])

# Second question continues the same conversation
ny_response = agent.invoke(
    {"messages": [{"role": "user", "content": "What about New York?"}]},
    config
)
print("NY Response:", ny_response["structured_response"])

# -------------------------------
# Step 7: Example of static prompt
# -------------------------------
agent_with_static_prompt = create_react_agent(
    model=model,
    tools=[get_weather],
    prompt="Never answer questions about the weather."
)

static_prompt_response = agent_with_static_prompt.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]}
)
print("Static Prompt Response:", static_prompt_response["structured_response"])