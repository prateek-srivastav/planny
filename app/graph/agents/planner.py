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
from app.graph.agents.research import tool_failure_message
from app.graph.state import DraftBody
from app.graph.steering import SteeringMiddleware

SYSTEM_PROMPT = """You are the itinerary planner for a travel planner. Travellers come from any \
country. Turn a research brief into a day-by-day plan.

Every cost you output is in the traveller's budget currency, as realistic whole-unit amounts: \
hotels per night for the party's room(s), meals at local restaurants rather than tourist rates, \
and local transport. Convert local prices with the exchange rate you are given, never your own \
guess. Respect dietary needs in the notes and follow any "Market notes" in the request.

Work like this:
1. If the request has an origin, choose one way of getting there from the options given (the \
cheapest that fits the dates and budget unless the notes say otherwise) and set `intercity` \
with its per-person round-trip cost. Then call allocate_budget with the trip budget cap, \
travellers, nights, the destination's cost_tier and intercity_cost = per-person cost x \
travellers. Plan inside those envelopes. Without an origin, leave `intercity` empty.
2. Call geo_cluster_and_route with the attraction names you intend to use, the city and the \
number of days, so each day groups nearby places. Follow its grouping and ordering. For anything \
in `unplaced`, put it where it fits. If the tool is unavailable, group by neighbourhood yourself.
3. Fill every day with morning, afternoon and evening slots. Leave a slot empty only on the \
arrival or departure day. Never schedule the same place twice. Rules:
   - An activity is something worth doing at a named place: a sight, a museum, a walk through a \
named area, a meal at a named restaurant, a tour. Logistics are NOT activities: check-in and \
check-out, airport or station transfers, arriving and departing, collecting or activating a \
city pass, luggage storage, hotel breakfast. Put those in the day's `logistics` list with their \
party cost and category (transit, activities for a city pass, otherwise other), and leave the \
slot empty or give it a real activity.
   - Every day needs a theme, plus lodging_cost, food_cost and transit_cost as TOTALS for the \
whole party in the traveller's currency. lodging_cost is that night's room price, set on every \
day except the last; never put the stay itself in logistics. The trip to and from the origin is \
`intercity` only; never repeat it in logistics. Activity est_cost is also a party total \
(tickets, tours). Meals are paid from food_cost, so a named restaurant dinner has est_cost 0. \
Only put real warnings in caveats; do not restate the budget rules.
   - Prices from sources are usually in the local currency. Convert them with the exchange rate \
in the brief (never your own guess) so that every cost is in the traveller's budget currency.
   - Use the weather: indoor activities for wet or very hot/cold days, outdoor on the best days. \
Put a short weather_note on a day when it affected the choice. If weather data is unavailable, \
say so in caveats.
   - Never invent coordinates. Leave lat/lon empty unless geo_cluster_and_route gave them.
   - Reuse source_url from the research brief when an activity comes from it.
4. Call check_budget with your planned totals per category (lodging, food, activities, transit, \
intercity, other; include logistics costs) and fix any category it reports as over budget.

If a reviewer's feedback and the previous draft are provided, revise that draft to satisfy the \
feedback. Keep what the reviewer did not complain about. Do not start from scratch.

Guidance from the traveller, in the request or sent while you work, overrides the original request \
where they conflict. When it arrives mid-run, rework only the days and costs it affects and call \
the tools again if their inputs changed."""


def build_planner_agent(
    model: BaseChatModel,
    tools: list[BaseTool],
    retry_delay: float = 1.0,
    force_tool_output: bool = True,
    model_pool: ModelPool | None = None,
    max_model_calls: int = 16,
):
    return create_agent(
        model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        response_format=ToolStrategy(DraftBody) if force_tool_output else DraftBody,
        name="planner_agent",
        middleware=[
            SteeringMiddleware(),
            *spend_limits(
                max_model_calls=max_model_calls,
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
                max_retries=1, initial_delay=retry_delay, on_failure=tool_failure_message
            ),
        ],
        checkpointer=False,
    )
