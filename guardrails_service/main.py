"""
Service 3 - Guardrails Service (AI Property Triage System)

FastAPI microservice that wraps NeMo Guardrails "self-checking rails":

    POST /check/input   -> is the text a genuine property listing?
    POST /check/output  -> is the AI-generated listing report safe to release?

Both endpoints accept  {"text": "..."}  and return
    {"pass": true|false, "reason": "...", "safe_text": "..."}

LLM routing: NeMo's 'openai' engine is pointed at Google's OpenAI-compatible
Gemini endpoint through environment variables (see the block right below).
"""
import os

from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------------------------------
# LLM routing (must run BEFORE NeMo / LangChain create their OpenAI client)
# -----------------------------------------------------------------------------
# Gemini's OpenAI-compatible endpoint. Override OPENAI_BASE_URL to use any other
# OpenAI-compatible server (vLLM, Ollama, LiteLLM, ...).
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Accept either OPENAI_API_KEY (as used by the OpenAI SDK) or GEMINI_API_KEY.
os.environ["OPENAI_API_KEY"] = os.getenv("GEMINI_API_KEY", "missing-api-key")
os.environ["OPENAI_BASE_URL"] = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
# Older langchain-openai releases read this legacy name instead.


import logging  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, Literal, Optional  # noqa: E402

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field  # noqa: E402

from nemoguardrails import LLMRails, RailsConfig  # noqa: E402
from nemoguardrails.llm.call import llm_call  # noqa: E402
from nemoguardrails.llm.types import Task  # noqa: E402

logger = logging.getLogger("guardrails-service")
logging.basicConfig(level=logging.INFO)

# The ./config directory that sits next to this file (config.yml, prompts.yml,
# rails.co). Resolved from __file__ so it works from any working directory.
CONFIG_DIR = Path(__file__).resolve().parent / "config"

# Prefix that the Colang refusal messages (rails.co) start with.
FAIL_PREFIX = "FAIL:"

app = FastAPI(
    title="Guardrails Service",
    description="Self-checking NeMo Guardrails for the AI Property Triage System.",
    version="1.0.0",
)

# Global LLMRails engine, created once on application startup.
rails: Optional[LLMRails] = None


# -----------------------------------------------------------------------------
# Pydantic schemas
# -----------------------------------------------------------------------------
class CheckRequest(BaseModel):
    """Input schema shared by both endpoints: {"text": "<text to check>"}."""

    text: str = Field(..., min_length=1, description="Text to check.")


class CheckResponse(BaseModel):
    """Output schema shared by both endpoints.

    `pass` is a reserved keyword in Python, so the attribute is called `is_pass`
    and exposed as "pass" in the JSON through an alias.
    """

    # populate_by_name lets us build the model with `is_pass=` in Python code.
    model_config = ConfigDict(populate_by_name=True)

    is_pass: bool = Field(..., alias="pass", description="True if the text is allowed.")
    reason: str = Field("none", description="Why the text was rejected ('none' if it passed).")
    safe_text: str = Field("", description="The original text if it passed, otherwise ''.")


# -----------------------------------------------------------------------------
# Self-check actions (yes = SAFE, no = UNSAFE)
# -----------------------------------------------------------------------------
# prompts.yml asks the model to answer "yes" when the text is SAFE. NeMo's
# built-in self_check_* actions use the opposite convention ("yes" = block) and
# newer releases return a RailOutcome object instead of a bool, which would not
# work with the `$allowed = execute ...; if not $allowed` logic of rails.co.
# These two actions replace the built-ins under the same names. Flows, prompt
# task names and the self-check architecture stay exactly the same.
#
# The verdict is parsed strictly and the checks FAIL CLOSED: an empty answer,
# an unexpected answer, or any LLM error (network, quota, Gemini safety block)
# counts as "unsafe".
_VERDICT_RE = re.compile(r"\W*(yes|no)\b", re.IGNORECASE)


def _verdict_is_safe(raw_answer: str) -> bool:
    """Return True only if the model's answer starts with 'yes'."""
    match = _VERDICT_RE.match(raw_answer or "")
    return bool(match) and match.group(1).lower() == "yes"


async def _run_self_check(llm_task_manager: Any, llm: Any, task: Task, prompt_vars: Dict[str, str]) -> bool:
    """Render the prompt for `task`, call the LLM and interpret the yes/no answer."""
    try:
        prompt = llm_task_manager.render_task_prompt(task=task, context=prompt_vars)
        response = await llm_call(llm, prompt)
        # Newer NeMo returns an LLMResponse object, older releases a plain string.
        answer = str(getattr(response, "content", response) or "").strip()
    except Exception:  # noqa: BLE001 - deliberately broad: guard must fail closed
        logger.exception("Self-check LLM call failed for task %s - failing closed.", task.value)
        return False

    logger.info("Self-check %s verdict: %r", task.value, answer[:80])
    return _verdict_is_safe(answer)


async def self_check_input(llm_task_manager: Any, context: Optional[dict] = None, llm: Any = None, **kwargs) -> bool:
    """Input rail action: True if the user text is a safe, genuine property listing."""
    text = (context or {}).get("user_message") or ""
    if not text.strip():
        return False
    return await _run_self_check(llm_task_manager, llm, Task.SELF_CHECK_INPUT, {"user_input": text})


async def self_check_output(llm_task_manager: Any, context: Optional[dict] = None, llm: Any = None, **kwargs) -> bool:
    """Output rail action: True if the AI-generated report is safe to release."""
    ctx = context or {}
    # `bot_response` is set explicitly by process_check(); `bot_message` is the
    # variable NeMo itself populates from the last assistant message.
    report = ctx.get("bot_response") or ctx.get("bot_message") or ""
    if not report.strip():
        return False
    return await _run_self_check(llm_task_manager, llm, Task.SELF_CHECK_OUTPUT, {"bot_response": report})


# -----------------------------------------------------------------------------
# Startup: build the global rails engine
# -----------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event() -> None:
    """Load ./config and create the global LLMRails engine (runs once)."""
    global rails

    config = RailsConfig.from_path(str(CONFIG_DIR))

    # Route the 'openai' engine to the endpoint chosen above. NeMo 0.24+ uses its
    # own OpenAI-compatible client, which ignores OPENAI_BASE_URL and reads the
    # endpoint from the model's `parameters.base_url` instead (OPENAI_API_KEY is
    # read from the environment as usual).
    # Optional: GUARDRAILS_LLM_MODEL swaps the Gemini model without editing config.yml.
    model_override = os.getenv("GUARDRAILS_LLM_MODEL")
    for model_cfg in config.models:
        if model_cfg.type == "main":
            model_cfg.parameters = {
                **(model_cfg.parameters or {}),
                "base_url": os.environ["OPENAI_BASE_URL"],
            }
            if model_override:
                model_cfg.model = model_override

    rails = LLMRails(config)
    rails.register_action(self_check_input, name="self_check_input")
    rails.register_action(self_check_output, name="self_check_output")
    logger.info("Guardrails engine ready (base URL: %s).", os.environ["OPENAI_BASE_URL"])


# -----------------------------------------------------------------------------
# Core processing
# -----------------------------------------------------------------------------
def _extract_bot_message(result: Any) -> str:
    """Normalise whatever generate_async returns into the bot message string."""
    response = getattr(result, "response", result)  # GenerationResponse -> .response
    if isinstance(response, list):  # list of messages -> last one
        response = response[-1] if response else ""
    if isinstance(response, dict):  # {"role": ..., "content": ...}
        response = response.get("content", "")
    return str(response or "").strip()


async def process_check(text: str, endpoint_type: Literal["input", "output"]) -> CheckResponse:
    """Run the input or output rails on `text` and map the result to the API schema."""
    if rails is None:
        raise HTTPException(status_code=503, detail="Guardrails engine is not initialised yet.")

    if endpoint_type == "input":
        # Input rails only: the text is treated as the user message.
        messages = [{"role": "user", "content": text}]
        options = {"rails": ["input"]}
    else:
        # Output rails only. generate_async() has no `context=` keyword, so the
        # bot response is injected with a "context" message instead, which is
        # NeMo's supported way of setting context variables. It is also sent as
        # the final assistant message, which is what output rails audit.
        messages = [
            {"role": "context", "content": {"bot_response": text}},
            {"role": "user", "content": "Generate the property listing report."},
            {"role": "assistant", "content": text},
        ]
        options = {"rails": ["output"]}

    try:
        result = await rails.generate_async(messages=messages, options=options)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Guardrails engine error")
        raise HTTPException(status_code=502, detail="Guardrails engine error.") from exc

    bot_message = _extract_bot_message(result)

    # A blocked text comes back as the Colang refusal message: "FAIL: <reason>".
    if bot_message.startswith(FAIL_PREFIX):
        reason = bot_message[len(FAIL_PREFIX):].strip() or "Rejected by policy."
        return CheckResponse(is_pass=False, reason=reason, safe_text="")

    return CheckResponse(is_pass=True, reason="none", safe_text=text)


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.post("/check/input", response_model=CheckResponse, response_model_by_alias=True)
async def check_input(request: CheckRequest) -> CheckResponse:
    """Reject spam, offensive, off-topic text and legal/tax advice requests."""
    return await process_check(request.text, "input")


@app.post("/check/output", response_model=CheckResponse, response_model_by_alias=True)
async def check_output(request: CheckRequest) -> CheckResponse:
    """Reject reports with false legal claims, invented prices or fake certifications."""
    return await process_check(request.text, "output")