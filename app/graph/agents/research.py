from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRetryMiddleware,
    ToolRetryMiddleware,
)
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from app.core.llm import should_retry_model
from app.core.model_pool import ModelPool
from app.graph.agents.limits import spend_limits
from app.graph.state import ResearchBody
from app.graph.steering import SteeringMiddleware
from app.tools import collector

SYSTEM_PROMPT = """You are the research agent for a travel planner. Travellers come from any \
country. Gather the facts a planner needs to build a good trip for *this* traveller. Do not write \
the itinerary.

The request names the traveller's nationality and budget currency. Use them: prefer sources from \
the traveller's home market for costs and booking (they quote prices in the right currency), \
and consider what matters to someone from there: which holidays and seasons overlap the trip, \
dietary needs in the notes, whether their cards or payment apps work at the destination, and the \
entry rules for their passport. Follow any "Market notes" in the request.

Work like this:
1. Call destination_conditions once for the trip dates. Work out the destination's local \
currency; if it differs from the budget currency, call exchange_rate (budget currency -> local \
currency) once.
2. Call web_search 3-6 times (a hard limit of 6 per run; after that the tool refuses, so stop \
and answer) with specific queries: top attractions and what makes them worth visiting, which \
areas to stay in, local food, getting around, rough prices, and anything tailored to the \
traveller's interests. Prefer recent sources.
3. Produce the structured brief. Rules:
   - attractions: 8-16 candidates, each a real, named place that a map would find. Include a mix \
of indoor and outdoor, and honour the traveller's interests. Give a realistic est_cost_per_person \
in the traveller's currency (0 if free).
   - cost_tier: the destination's overall price level for a visitor.
   - source_urls: only URLs that web_search actually returned and that you relied on.
   - entry_requirements: the traveller's nationality is given. Search for the passport, visa (or \
e-visa / visa-free stay) and health rules for *that nationality* entering the destination, \
preferring official government or embassy pages, and summarise them with the permitted stay. \
If you could not verify them, write "not confirmed" instead of guessing. If no nationality is \
given, leave it empty.
   - local_currency: ISO 4217 code of the destination.
   - getting_there: only if the request says where the party travels from. Spend one search on \
it, then list 1-3 realistic options (flight, train, bus, ...) from the origin to the destination \
and back for the trip dates, each with a per-person round-trip cost in the budget currency and \
a one-way door-to-door duration. Without an origin, leave it empty.
   - If a tool reports it is unavailable, carry on with what you know and do not invent data.

Be factual and concise. If a reviewer's feedback is provided, address it directly. Guidance from \
the traveller, in the request or sent while you work, overrides the original request where they \
conflict: search again for whatever it changes."""


def tool_failure_message(exc: Exception) -> str:
    """Called when a tool still fails after its retries. Record it and let the agent carry on."""
    collector.record_degraded(str(exc) or type(exc).__name__)
    return f"TOOL UNAVAILABLE: {exc}. Continue without it and do not make up its data."


def build_research_agent(
    model: BaseChatModel,
    tools: list[BaseTool],
    retry_delay: float = 1.0,
    force_tool_output: bool = True,
    model_pool: ModelPool | None = None,
    max_model_calls: int = 16,
    max_searches: int | None = None,
):
    return create_agent(
        model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        response_format=ToolStrategy(ResearchBody) if force_tool_output else ResearchBody,
        name="research_agent",
        middleware=[
            SteeringMiddleware(),
            *spend_limits(
                max_model_calls=max_model_calls,
                search_tool="web_search",
                max_searches=max_searches,
            ),
            # Outermost: once a model has used up its own retries, rotate to the next one.
            *([model_pool] if model_pool else []),
            # on_failure="error": raise once retries are spent, so the fallback above can react (and
            # so a dead model fails the plan with its real error, not a missing structured output).
            ModelRetryMiddleware(
                max_retries=2 if model_pool else 4,
                initial_delay=retry_delay,
                max_delay=30.0,
                on_failure="error",
                retry_on=should_retry_model,
            ),
            ToolRetryMiddleware(
                max_retries=2, initial_delay=retry_delay, on_failure=tool_failure_message
            ),
        ],
        checkpointer=False,  # never inherit the parent's checkpointer: replays must start fresh
    )
