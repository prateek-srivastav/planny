"""Web search behind a small protocol so the provider is a config choice (SEARCH_PROVIDER)."""

from typing import Protocol

import httpx
from pydantic import BaseModel

from app.core.config import Settings
from app.tools.collector import ToolFailure

DEFAULT_RESULT_CHARS = 600


class SearchResult(BaseModel):
    title: str
    url: str
    published_date: str | None = None
    text: str = ""


class SearchProvider(Protocol):
    name: str

    async def search(
        self,
        query: str,
        num_results: int = 5,
        objective: str | None = None,
        country: str | None = None,
    ) -> list[SearchResult]: ...


class ExaSearch:
    """Exa `/search` with `contents.highlights`.

    Highlights are the passages most relevant to the query, so each result is small and on-topic
    (a page's first N characters, `contents.text`, is often navigation and boilerplate). They come
    back in the same call, so there is no separate scrape step. `maxCharacters` is set because the
    uncapped default is several thousand characters per result, which would swamp a small context.
    `objective` tells Exa what the search is for, beyond the literal query.
    """

    name = "exa"

    def __init__(
        self,
        client: httpx.AsyncClient,
        api_key: str | None,
        country: str | None = None,
        result_chars: int = DEFAULT_RESULT_CHARS,
    ):
        self._client, self._key, self._country = client, api_key, country
        self._chars = result_chars

    async def search(
        self,
        query: str,
        num_results: int = 5,
        objective: str | None = None,
        country: str | None = None,
    ) -> list[SearchResult]:
        if not self._key:
            raise ToolFailure("web_search: EXA_API_KEY is not set", retryable=False)
        body: dict = {
            "query": query,
            "numResults": num_results,
            "type": "auto",
            "contents": {"highlights": {"maxCharacters": self._chars}},
        }
        if objective:
            body["objective"] = objective[:4096]
        market = country or self._country
        if market:
            body["userLocation"] = market  # localise results to the traveller's market
        resp = await self._client.post(
            "https://api.exa.ai/search", headers={"x-api-key": self._key}, json=body
        )
        _check(resp, "exa")
        return [
            SearchResult(
                title=r.get("title") or r["url"],
                url=r["url"],
                published_date=r.get("publishedDate"),
                text=" … ".join(r.get("highlights") or [])[: self._chars],
            )
            for r in resp.json().get("results", [])
        ]


class SerperSearch:
    """Serper.dev (Google results). Snippets only, no page text."""

    name = "serper"

    def __init__(
        self,
        client: httpx.AsyncClient,
        api_key: str | None,
        country: str | None = None,
        result_chars: int = DEFAULT_RESULT_CHARS,
    ):
        self._client, self._key, self._country = client, api_key, country
        self._chars = result_chars

    async def search(
        self,
        query: str,
        num_results: int = 5,
        objective: str | None = None,
        country: str | None = None,
    ) -> list[SearchResult]:
        if not self._key:
            raise ToolFailure("web_search: SERPER_API_KEY is not set", retryable=False)
        resp = await self._client.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": self._key},
            json={
                "q": query,
                "num": num_results,
                **(
                    {"gl": (country or self._country).lower()} if (country or self._country) else {}
                ),
            },
        )
        _check(resp, "serper")
        return [
            SearchResult(
                title=r.get("title", r["link"]),
                url=r["link"],
                published_date=r.get("date"),
                text=(r.get("snippet") or "")[: self._chars],
            )
            for r in resp.json().get("organic", [])
        ]


def _check(resp: httpx.Response, provider: str) -> None:
    code = resp.status_code
    if code < 400:
        return
    # 401/402/403 (bad key, out of credits, forbidden) and other 4xx except 408/429 won't fix
    # themselves; 429 and 5xx will, so those stay retryable.
    reason = {401: "invalid API key", 402: "out of credits", 403: "access denied"}.get(code, "")
    retryable = code in (408, 429) or code >= 500
    raise ToolFailure(
        f"web_search: {provider} returned HTTP {code}" + (f" ({reason})" if reason else ""),
        retryable=retryable,
    )


def make_provider(settings: Settings, client: httpx.AsyncClient) -> SearchProvider:
    if settings.search_provider == "serper":
        return SerperSearch(
            client, settings.serper_api_key, settings.search_country, settings.search_result_chars
        )
    return ExaSearch(
        client, settings.exa_api_key, settings.search_country, settings.search_result_chars
    )
