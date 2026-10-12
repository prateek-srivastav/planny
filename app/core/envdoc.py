"""Diagnose a `.env` file without ever printing a secret.

Most "it says no API key" problems are a typo in the file, a key on the wrong line, or a shell
variable overriding it. This reports those, showing only lengths and the well-known public prefix
of each provider's key format (e.g. `sk-or-`), never key material.
"""

import difflib
import os
import re
from dataclasses import dataclass
from pathlib import Path

# variable -> (provider, expected key prefix or None)
KEY_VARS: dict[str, tuple[str, str | None]] = {
    "OPENROUTER_API_KEY": ("OpenRouter", "sk-or-"),
    "GROQ_API_KEY": ("Groq", "gsk_"),
    "EXA_API_KEY": ("Exa", None),
    "SERPER_API_KEY": ("Serper", None),
    "GOOGLE_API_KEY": ("Google AI Studio", None),  # several key formats in use
    "ANTHROPIC_API_KEY": ("Anthropic", "sk-ant-"),
    "OPENAI_API_KEY": ("OpenAI", "sk-"),
}
PLACEHOLDER = re.compile(r"your|xxx|here|<|>|\.\.\.|changeme|example|todo", re.I)
_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$")


def obviously_invalid(var: str, value: str | None) -> bool:
    """A value that cannot be a real key for `var`: empty, a placeholder, wrong format, spaces."""
    if not value or not value.strip():
        return True
    prefix = KEY_VARS.get(var, (var, None))[1]
    return bool(
        PLACEHOLDER.search(value)
        or any(c.isspace() for c in value)
        or (prefix and not value.startswith(prefix))
    )


def reconcile_shell_overrides(env_file: Path, environ: dict[str, str] | None = None) -> list[str]:
    """Let a valid key in .env beat an obviously broken one exported in the shell.

    Normally an exported variable wins over .env (so you can override on purpose). But a stale
    placeholder like OPENROUTER_API_KEY=YOUR_KEY left in ~/.zshrc would silently break a correct
    .env, so in that one case the file wins. Returns a message per override applied.
    """
    from dotenv import dotenv_values

    environ = os.environ if environ is None else environ
    if not env_file.exists():
        return []
    file_values = dotenv_values(env_file)
    notes = []
    for var in KEY_VARS:
        shell, from_file = environ.get(var), _clean(file_values.get(var) or "")
        if shell is None or shell == from_file or not from_file:
            continue
        if obviously_invalid(var, shell) and not obviously_invalid(var, from_file):
            environ[var] = from_file
            notes.append(
                f"{var} is exported in your shell as an invalid value ({len(shell)} characters, "
                f"starts '{shell[:4]}...'); using the valid key from .env instead. "
                f"Remove it from your shell profile (~/.zshrc) to silence this."
            )
    return notes


@dataclass
class Finding:
    level: str  # "ok" | "warn" | "error"
    message: str


def _clean(raw: str) -> str:
    v = raw.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        v = v[1:-1]
    elif " #" in v:  # inline comment on an unquoted value
        v = v.split(" #", 1)[0].strip()
    return v.strip()


def diagnose(
    path: Path,
    environ: dict[str, str] | None = None,
    needed: list[str] | None = None,
    optional: dict[str, str] | None = None,
):
    """Return findings for `path`.

    `needed` = key variables that must be set. `optional` maps key variables that may be left
    blank to what happens without them; a blank one is a note, but a malformed value is still
    reported like any other.
    """
    environ = os.environ if environ is None else environ
    needed = needed if needed is not None else ["OPENROUTER_API_KEY"]
    optional = optional or {}
    needed = [*needed, *(v for v in optional if v not in needed)]
    out: list[Finding] = []
    if not path.exists():
        out.append(Finding("error", f"no .env file at {path} (copy .env.example to .env there)"))
        defined: dict[str, list[tuple[int, str]]] = {}
    else:
        data = path.read_bytes()
        if data.startswith(b"\xef\xbb\xbf"):
            out.append(Finding("error", ".env starts with a hidden BOM; re-save it as plain UTF-8"))
        if b"\r\n" in data:
            out.append(Finding("warn", ".env has Windows line endings (harmless here, but odd)"))
        out.append(Finding("ok", f"found {path}"))
        defined = {}
        for n, line in enumerate(data.decode("utf-8-sig", errors="replace").splitlines(), 1):
            if line.strip().startswith("#"):
                continue
            m = _LINE.match(line)
            if m:
                defined.setdefault(m.group(1), []).append((n, m.group(2)))

    for var, entries in defined.items():
        if len(entries) > 1:
            lines = ", ".join(str(n) for n, _ in entries)
            out.append(
                Finding(
                    "warn", f"{var}: defined {len(entries)} times (lines {lines}); the last wins"
                )
            )

    for var in needed:
        provider, prefix = KEY_VARS.get(var, (var, None))
        entries = defined.get(var, [])
        env_val = environ.get(var)

        if not entries:
            # A real key name (GROQ_API_KEY) is never a misspelling of another (GOOGLE_API_KEY).
            candidates = [d for d in defined if d not in KEY_VARS and d not in needed]
            near = difflib.get_close_matches(var, candidates, n=1, cutoff=0.7)
            hint = f" (did you mean to rename {near[0]} to {var}?)" if near else ""
            if env_val:
                out.append(Finding("ok", f"{var}: not in .env, but set in your shell environment"))
            elif var in optional:
                out.append(Finding("ok", f"{var}: not set ({optional[var]}){hint}"))
            else:
                out.append(Finding("error", f"{var}: not set in .env or the environment{hint}"))
            continue

        line_no, raw = entries[-1]
        value = _clean(raw)

        if raw != raw.strip() and raw.strip() == "":
            out.append(Finding("error", f"{var} (line {line_no}): only whitespace after '='"))
        elif not value and var in optional:
            out.append(Finding("ok", f"{var} (line {line_no}): blank ({optional[var]})"))
        elif not value:
            out.append(
                Finding(
                    "error", f"{var} (line {line_no}): empty. Paste your {provider} key after '='"
                )
            )
        else:
            if PLACEHOLDER.search(value):
                out.append(
                    Finding(
                        "error", f"{var} (line {line_no}): looks like a placeholder, not a real key"
                    )
                )
            elif prefix and not value.startswith(prefix):
                swapped = [
                    KEY_VARS[v][0]
                    for v, (_, p) in KEY_VARS.items()
                    if p and p != prefix and value.startswith(p)
                ]
                why = f"; that looks like a {swapped[0]} key (keys swapped?)" if swapped else ""
                out.append(
                    Finding(
                        "warn",
                        f"{var} (line {line_no}): {provider} keys start with '{prefix}'{why}",
                    )
                )
            else:
                out.append(Finding("ok", f"{var} (line {line_no}): set, {len(value)} characters"))
            if any(c.isspace() for c in value):
                out.append(Finding("error", f"{var} (line {line_no}): the key contains a space"))

        if env_val is not None and env_val != value:
            state = "empty" if not env_val else f"{len(env_val)} characters"
            out.append(
                Finding(
                    "warn",
                    f"{var}: your shell also exports it ({state}) and that value is used instead "
                    f"of .env. Run `unset {var}` and try again",
                )
            )
    return out
