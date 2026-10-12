"""The .env doctor: finds the usual mistakes, never prints key material."""

import os
from pathlib import Path

import pytest

from app.core.config import ENV_FILE, PROJECT_ROOT, Settings
from app.core.envdoc import diagnose, obviously_invalid, reconcile_shell_overrides

GOOD_OR = "sk-or-v1-" + "a" * 40
GOOD_GROQ = "gsk_" + "b" * 40


def run(tmp_path: Path, text: str, needed=("OPENROUTER_API_KEY",), environ=None):
    p = tmp_path / ".env"
    p.write_text(text)
    return diagnose(p, environ or {}, list(needed))


def messages(findings, level=None):
    return [f.message for f in findings if level is None or f.level == level]


def test_a_healthy_file_has_no_errors_or_warnings(tmp_path):
    f = run(tmp_path, f"OPENROUTER_API_KEY={GOOD_OR}\nMODEL=x\n")
    assert [m for m in messages(f) if m.startswith("OPENROUTER")] == [
        f"OPENROUTER_API_KEY (line 1): set, {len(GOOD_OR)} characters"
    ]
    assert not messages(f, "error") and not messages(f, "warn")


def test_missing_file_says_where_it_looked(tmp_path):
    f = diagnose(tmp_path / ".env", {}, ["OPENROUTER_API_KEY"])
    assert "no .env file at" in messages(f, "error")[0]


@pytest.mark.parametrize(
    "line", ["OPENROUTER_API_KEY=", "OPENROUTER_API_KEY=   ", 'OPENROUTER_API_KEY=""']
)
def test_empty_key_is_an_error(tmp_path, line):
    assert any(
        "empty" in m or "whitespace" in m for m in messages(run(tmp_path, line + "\n"), "error")
    )


def test_placeholder_is_an_error(tmp_path):
    f = run(tmp_path, "OPENROUTER_API_KEY=your_openrouter_key_here\n")
    assert any("placeholder" in m for m in messages(f, "error"))


def test_swapped_keys_are_named(tmp_path):
    f = run(tmp_path, f"OPENROUTER_API_KEY={GOOD_GROQ}\n")
    warn = " ".join(messages(f, "warn"))
    assert "start with 'sk-or-'" in warn and "looks like a Groq key" in warn


def test_quotes_are_tolerated_but_an_inner_space_is_not(tmp_path):
    assert not messages(run(tmp_path, f'OPENROUTER_API_KEY="{GOOD_OR}"\n'), "error")
    f = run(tmp_path, "OPENROUTER_API_KEY=sk-or-v1-abc def\n")
    assert any("contains a space" in m for m in messages(f, "error"))


def test_duplicate_definitions_warn_and_the_last_wins(tmp_path):
    f = run(tmp_path, f"OPENROUTER_API_KEY={GOOD_OR}\nOPENROUTER_API_KEY=\n")
    assert any("defined 2 times" in m for m in messages(f, "warn"))
    assert any("empty" in m for m in messages(f, "error"))  # the later, empty line is what counts


def test_misspelled_variable_name_is_suggested(tmp_path):
    f = run(tmp_path, f"OPENROUTER_API_KY={GOOD_OR}\n")
    assert any("rename OPENROUTER_API_KY" in m for m in messages(f, "error"))


def test_comments_and_export_prefix_are_understood(tmp_path):
    f = run(tmp_path, f"# OPENROUTER_API_KEY=old\nexport OPENROUTER_API_KEY={GOOD_OR}  # mine\n")
    assert not messages(f, "error") and not messages(f, "warn")


def test_a_shell_variable_that_overrides_the_file_is_flagged(tmp_path):
    f = run(
        tmp_path,
        f"OPENROUTER_API_KEY={GOOD_OR}\n",
        environ={"OPENROUTER_API_KEY": ""},
    )
    assert any("shell also exports it (empty)" in m and "unset" in m for m in messages(f, "warn"))


def test_key_only_in_the_shell_is_fine(tmp_path):
    f = run(tmp_path, "MODEL=x\n", environ={"OPENROUTER_API_KEY": GOOD_OR})
    assert not messages(f, "error")


def test_the_report_never_contains_key_material(tmp_path):
    f = run(
        tmp_path,
        f"OPENROUTER_API_KEY={GOOD_OR}\nGROQ_API_KEY=oops\n",
        ("OPENROUTER_API_KEY", "GROQ_API_KEY"),
    )
    report = " ".join(messages(f))
    assert "a" * 10 not in report and GOOD_OR not in report and "oops" not in report


# ---- Settings: where .env is read from, and key cleanup


def test_env_file_is_anchored_to_the_project_not_the_current_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert ENV_FILE == PROJECT_ROOT / ".env" and ENV_FILE.is_absolute()
    assert Settings.model_config["env_file"] == ENV_FILE


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (f'  "{GOOD_OR}"  ', GOOD_OR),
        (f"'{GOOD_OR}'", GOOD_OR),
        ("   ", None),
        ('""', None),
        ("", None),
    ],
)
def test_keys_are_cleaned_and_blank_means_unset(raw, expected):
    assert Settings(_env_file=None, openrouter_api_key=raw).openrouter_api_key == expected


def test_a_blank_key_in_the_environment_is_unset_not_an_empty_header(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "  ")
    assert Settings(_env_file=None).openrouter_api_key is None
    assert os.environ["OPENROUTER_API_KEY"] == "  "


# ---- a broken shell variable must not defeat a good .env


def env_file(tmp_path, text):
    p = tmp_path / ".env"
    p.write_text(text)
    return p


def test_placeholder_in_the_shell_loses_to_a_valid_key_in_the_file(tmp_path):
    """A placeholder exported by the shell profile must not override a valid key in .env."""
    environ = {"OPENROUTER_API_KEY": "YOUR_OPENROUTER_API_KEY"}
    notes = reconcile_shell_overrides(
        env_file(tmp_path, f"OPENROUTER_API_KEY={GOOD_OR}\n"), environ
    )
    assert environ["OPENROUTER_API_KEY"] == GOOD_OR
    assert len(notes) == 1 and "starts 'YOUR..." in notes[0] and GOOD_OR not in notes[0]


@pytest.mark.parametrize("bad", ["", "  ", "your-key-here", GOOD_GROQ, "sk-or-v1-has space"])
def test_every_kind_of_broken_shell_value_is_overridden(tmp_path, bad):
    environ = {"OPENROUTER_API_KEY": bad}
    reconcile_shell_overrides(env_file(tmp_path, f"OPENROUTER_API_KEY={GOOD_OR}\n"), environ)
    assert environ["OPENROUTER_API_KEY"] == GOOD_OR


def test_a_valid_shell_key_still_wins_over_the_file(tmp_path):
    other = "sk-or-v1-" + "c" * 40
    environ = {"OPENROUTER_API_KEY": other}  # deliberate override: respected
    assert (
        reconcile_shell_overrides(env_file(tmp_path, f"OPENROUTER_API_KEY={GOOD_OR}\n"), environ)
        == []
    )
    assert environ["OPENROUTER_API_KEY"] == other


def test_nothing_changes_when_the_file_value_is_also_bad_or_absent(tmp_path):
    environ = {"OPENROUTER_API_KEY": "YOUR_KEY"}
    assert reconcile_shell_overrides(env_file(tmp_path, "OPENROUTER_API_KEY=\n"), environ) == []
    assert reconcile_shell_overrides(env_file(tmp_path, "MODEL=x\n"), environ) == []
    assert reconcile_shell_overrides(tmp_path / "missing.env", environ) == []
    assert environ["OPENROUTER_API_KEY"] == "YOUR_KEY"


def test_obviously_invalid():
    assert obviously_invalid("OPENROUTER_API_KEY", None) and obviously_invalid(
        "GROQ_API_KEY", "x y"
    )
    assert not obviously_invalid("OPENROUTER_API_KEY", GOOD_OR)
    assert not obviously_invalid("EXA_API_KEY", "34cce318-0000-4000-8000-123456789abc")


def test_any_duplicated_setting_is_flagged(tmp_path):
    f = run(tmp_path, f"OPENROUTER_API_KEY={GOOD_OR}\nMODEL_FALLBACKS=a\nMODEL_FALLBACKS=b\n")
    assert any("MODEL_FALLBACKS: defined 2 times (lines 2, 3)" in m for m in messages(f, "warn"))


def test_optional_keys_may_be_blank_but_not_malformed(tmp_path):
    p = tmp_path / ".env"
    p.write_text("OPENROUTER_API_KEY=\nGROQ_API_KEY=your_key_here\n")
    optional = {k: "its models are skipped" for k in ("OPENROUTER_API_KEY", "GROQ_API_KEY")}
    optional["GOOGLE_API_KEY"] = "its models are skipped"
    f = diagnose(p, {}, [], optional)
    assert "OPENROUTER_API_KEY (line 1): blank (its models are skipped)" in messages(f, "ok")
    assert "GOOGLE_API_KEY: not set (its models are skipped)" in messages(f, "ok")
    assert messages(f, "error") == [
        "GROQ_API_KEY (line 2): looks like a placeholder, not a real key"
    ]
