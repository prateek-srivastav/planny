from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.core.llm import (
    classifier_structured_kwargs,
    forces_tool_calling,
    make_chat_model,
    resolve_models,
)
from app.core.model_pool import ModelPool
from app.graph.nodes import build_nodes
from app.graph.state import PlanState
from app.graph.steering import SteeringDesk
from app.tools.registry import Deps


def build_graph(
    checkpointer: BaseCheckpointSaver,
    deps: Deps,
    model: BaseChatModel | None = None,
    classifier: BaseChatModel | None = None,
    fallback_models: list[BaseChatModel] | None = None,
    steering: SteeringDesk | None = None,
):
    """Wire the workflow.

        validate -> research_agent -> planner_agent -> human_review
        human_review --approve-> finalize -> END
                     --modify--> apply_patch -> human_review
                     --reject--> route_feedback -> research_agent | planner_agent

    Every node except `finalize` routes with `Command(goto=...)`, so there is no routing table to
    drift out of sync and an error can short-circuit to END (static edges would still fire).
    """
    # Injected models (tests) use the same forced-tool path as the free routers.
    # Injected models (tests) are explicit wiring: don't also build settings-based fallbacks.
    # Models whose provider key is not set are left out (see resolve_models).
    chain = resolve_models(deps.settings) if model is None else None
    use_settings_fallbacks = fallback_models is None and chain is not None
    fallback_specs = chain.fallbacks if use_settings_fallbacks else []
    force_tool_output = (
        chain is None
        or forces_tool_calling(chain.primary)
        or any(forces_tool_calling(s) for s in fallback_specs)
    )
    classifier, classifier_kwargs = build_classifier(deps, classifier)
    model = model or make_chat_model(chain.primary, deps.settings)
    if fallback_models is None:
        fallback_models = [make_chat_model(s, deps.settings) for s in fallback_specs]
    pool = (
        ModelPool(
            fallback_models,
            deps.settings.model_cooldown_seconds,
            quota_cooldown_seconds=deps.settings.model_quota_cooldown_seconds,
        )
        if fallback_models
        else None
    )
    nodes = build_nodes(
        model, classifier, deps, force_tool_output, classifier_kwargs, pool, steering
    )

    g = StateGraph(PlanState)
    for name, fn in nodes.items():
        g.add_node(name, fn)
    g.add_edge(START, "validate")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)


def build_classifier(
    deps: Deps, classifier: BaseChatModel | None = None
) -> tuple[BaseChatModel, dict]:
    """The small model for routing decisions, and the kwargs for its structured output."""
    if classifier is not None:
        return classifier, {"method": "function_calling"}
    spec = resolve_models(deps.settings).classifier
    return make_chat_model(spec, deps.settings), classifier_structured_kwargs(spec)
