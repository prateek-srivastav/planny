"""Hard caps on how much an agent may spend in one run.

Bounds the cost of a model that keeps calling tools instead of answering. Two layers:

1. A per-tool cap (`web_search`): further calls return an error telling the agent to stop
   researching and answer, so it can still finish with a usable result.
2. A model-call cap that *raises*: if a model ignores layer 1, the plan fails fast with a clear
   message rather than burning quota indefinitely.
"""

from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware


def spend_limits(
    *, max_model_calls: int, search_tool: str | None = None, max_searches: int | None = None
) -> list:
    middleware = []
    if search_tool and max_searches is not None:
        middleware.append(
            ToolCallLimitMiddleware(
                tool_name=search_tool, run_limit=max_searches, exit_behavior="continue"
            )
        )
    middleware.append(ModelCallLimitMiddleware(run_limit=max_model_calls, exit_behavior="error"))
    return middleware
