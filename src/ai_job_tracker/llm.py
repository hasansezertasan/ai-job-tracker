"""LLM client for job analysis via pydantic-ai."""

from pydantic_ai import Agent
from pydantic_ai.models import infer_model
from pydantic_ai.providers import infer_provider_class

from ai_job_tracker.config import AnalysisResult, PROMPT_TEMPLATE, settings


async def analyze_job_posting(prompt: str) -> dict:
    """Submit a prompt to the configured LLM and return structured analysis dict."""
    if not settings.ai_api_key:
        raise RuntimeError(
            "AI_API_KEY is required. Set it in .env — see .env.example for provider-specific examples."
        )
    model = infer_model(
        settings.ai_model,
        provider_factory=lambda name: infer_provider_class(name)(api_key=settings.ai_api_key),
    )
    agent = Agent(model, output_type=AnalysisResult)
    result = await agent.run(prompt)
    return result.output.model_dump()


def _description_to_text(value) -> str:
    """Coerce a description value to a string suitable for slicing.

    NaN values (floats) from pandas can sneak into job dicts; treat any
    non-string as missing so we don't crash on `value[:2000]`.
    """
    if value is None:
        return "N/A"
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        return "N/A"
    return str(value)


def build_prompt(profile: str, job: dict) -> str:
    """Build prompt from profile and job data."""
    return PROMPT_TEMPLATE.format(
        profile=profile,
        title=job.get("title", "N/A") or "N/A",
        company=job.get("company", "N/A") or "N/A",
        location=job.get("location", "N/A") or "N/A",
        url=job.get("job_url", "N/A") or "N/A",
        description=_description_to_text(job.get("description"))[:2000],
    )
