"""
Generator module for Blogger Engine RAG.

Provides a provider-agnostic LLM client (Groq, Gemini, OpenAI, or custom OpenAI-compatible endpoints)
configured via environment variables. Assembles grounded prompts with strict citation constraints,
and enforces confidence thresholds to refuse ungrounded queries.
"""

import os
import re
import time
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import httpx

from app.core.retriever import SearchResult

logger = logging.getLogger("blogger_engine.generator")

# Default settings
DEFAULT_PROVIDER = "groq"
DEFAULT_MODEL = "llama-3.1-8b-instant"
DEFAULT_SIMILARITY_THRESHOLD = 0.65

PROVIDER_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
}


@dataclass
class Citation:
    doc_tag: str
    chunk_id: str
    title: str
    url: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GeneratedAnswer:
    answer: str
    citations: List[Citation]
    sources: List[Dict[str, Any]]
    confidence_score: float
    refused: bool
    latency_ms: float
    model: str
    provider: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "sources": self.sources,
            "confidence_score": self.confidence_score,
            "refused": self.refused,
            "latency_ms": self.latency_ms,
            "model": self.model,
            "provider": self.provider,
        }


class AnswerGenerator:
    """Provider-agnostic LLM generator with grounded prompt assembly and citation tracking."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
        timeout: float = 30.0
    ):
        raw_provider = provider if provider is not None else os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER)
        self.provider = raw_provider.strip().lower()

        raw_model = model if model is not None else os.getenv("LLM_MODEL", DEFAULT_MODEL)
        self.model = raw_model.strip()

        raw_api_key = api_key if api_key is not None else os.getenv("LLM_API_KEY", "")
        self.api_key = raw_api_key.strip()

        # Determine API base URL
        raw_base_url = (base_url if base_url is not None else os.getenv("LLM_BASE_URL", "")).strip()
        if raw_base_url:
            self.base_url = raw_base_url.rstrip("/")
        else:
            default_base_url = PROVIDER_BASE_URLS.get(self.provider, "https://api.groq.com/openai/v1")
            self.base_url = default_base_url.strip().rstrip("/")

        raw_thresh = similarity_threshold if similarity_threshold is not None else os.getenv("SIMILARITY_THRESHOLD")
        self.similarity_threshold = float(raw_thresh) if raw_thresh is not None else DEFAULT_SIMILARITY_THRESHOLD
        self.timeout = timeout

    def __repr__(self) -> str:
        has_key = "Yes" if bool(self.api_key) else "No"
        return f"<AnswerGenerator provider={self.provider} model={self.model} base_url={self.base_url} api_key_set={has_key}>"

    def build_system_prompt(self) -> str:
        """Create strict instructions requiring grounding and citations."""
        return (
            "You are an expert technical QA assistant for the Blogger Engine RAG system.\n"
            "Your task is to answer the user's question accurately using ONLY the provided source passages.\n\n"
            "Strict Grounding Rules:\n"
            "1. Answer based solely on the provided source passages. Do NOT extrapolate or assume facts not present in the text.\n"
            "2. For every factual claim you make, cite the supporting source passage tag (e.g. [Doc 1], [Doc 2]) immediately after the claim.\n"
            "3. If the provided passages do not contain sufficient information to answer the question, you MUST refuse and state:\n"
            '   "I do not have enough information in the provided blog articles to answer this question."\n'
            "4. Never invent facts, links, or citations not explicitly in the context.\n"
            "5. Provide a direct, well-structured, and concise answer."
        )

    def build_user_prompt(self, query: str, passages: List[SearchResult]) -> str:
        """Format the retrieved passages into numbered context tags."""
        context_blocks = []
        for i, p in enumerate(passages, start=1):
            block = (
                f"[Doc {i}] Title: {p.title}\n"
                f"URL: {p.url}\n"
                f"Content: {p.content.strip()}"
            )
            context_blocks.append(block)

        context_str = "\n\n".join(context_blocks)
        return (
            f"Source Passages:\n"
            f"--------------------\n"
            f"{context_str}\n"
            f"--------------------\n\n"
            f"User Question: {query}\n\n"
            f"Answer (with [Doc X] citations):"
        )

    def extract_citations(self, text: str, passages: List[SearchResult]) -> List[Citation]:
        """Parse [Doc X] references in the generated text and link to SearchResult objects."""
        # Find all cited indices e.g. [Doc 1], [Doc 2]
        matches = re.findall(r"\[Doc\s*(\d+)\]", text, re.IGNORECASE)
        cited_indices = set(int(m) for m in matches)

        citations = []
        for idx in sorted(cited_indices):
            # Passages are 1-indexed
            if 1 <= idx <= len(passages):
                p = passages[idx - 1]
                citations.append(
                    Citation(
                        doc_tag=f"[Doc {idx}]",
                        chunk_id=p.chunk_id,
                        title=p.title,
                        url=p.url,
                        score=p.score
                    )
                )
        return citations

    def generate_answer(
        self,
        query: str,
        passages: List[SearchResult],
        similarity_threshold: Optional[float] = None
    ) -> GeneratedAnswer:
        """
        Generate a cited answer for the query using retrieved passages.
        Refuses to answer if top passage similarity is below threshold.

        Args:
            query: The user query string.
            passages: Retrieved passages with similarity scores.
            similarity_threshold: Optional per-request threshold override.
        """
        start_time = time.time()
        sources_summary = [p.to_dict() for p in passages]

        effective_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self.similarity_threshold
        )

        # Check refusal condition 1: No passages retrieved
        if not passages:
            elapsed = (time.time() - start_time) * 1000
            return GeneratedAnswer(
                answer="I could not find any relevant blog articles matching your question.",
                citations=[],
                sources=[],
                confidence_score=0.0,
                refused=True,
                latency_ms=round(elapsed, 2),
                model=self.model,
                provider=self.provider
            )

        top_score = max(p.score for p in passages)

        # Check refusal condition 2: Below confidence threshold
        if top_score < effective_threshold:
            elapsed = (time.time() - start_time) * 1000
            logger.info(
                f"Query refused: top similarity {top_score:.3f} is below threshold {effective_threshold:.3f}"
            )
            return GeneratedAnswer(
                answer=(
                    f"I do not have enough confidence in the indexed blog articles to answer this question "
                    f"(top similarity: {top_score:.2f} is below required threshold: {effective_threshold:.2f})."
                ),
                citations=[],
                sources=sources_summary,
                confidence_score=round(top_score, 4),
                refused=True,
                latency_ms=round(elapsed, 2),
                model=self.model,
                provider=self.provider
            )

        # Check if API key is provided
        if not self.api_key:
            elapsed = (time.time() - start_time) * 1000
            return GeneratedAnswer(
                answer=(
                    "[API Key Not Configured] To generate live LLM responses, set LLM_API_KEY in your .env file. "
                    f"Relevant passages were successfully retrieved (top similarity: {top_score:.3f})."
                ),
                citations=[],
                sources=sources_summary,
                confidence_score=round(top_score, 4),
                refused=False,
                latency_ms=round(elapsed, 2),
                model=self.model,
                provider=self.provider
            )

        # Call LLM via standard OpenAI-compatible completions API
        system_prompt = self.build_system_prompt()
        user_prompt = self.build_user_prompt(query, passages)

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 1024
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                raw_text = data["choices"][0]["message"]["content"].strip()

            citations = self.extract_citations(raw_text, passages)
            elapsed = (time.time() - start_time) * 1000

            return GeneratedAnswer(
                answer=raw_text,
                citations=citations,
                sources=sources_summary,
                confidence_score=round(top_score, 4),
                refused=False,
                latency_ms=round(elapsed, 2),
                model=self.model,
                provider=self.provider
            )

        except Exception as exc:
            elapsed = (time.time() - start_time) * 1000
            if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
                status_code = exc.response.status_code
                error_detail = ""
                try:
                    err_json = exc.response.json()
                    if isinstance(err_json, dict):
                        err_obj = err_json.get("error")
                        if isinstance(err_obj, dict):
                            error_detail = err_obj.get("message") or str(err_obj)
                        elif isinstance(err_obj, str):
                            error_detail = err_obj
                        elif "message" in err_json:
                            error_detail = str(err_json["message"])
                except Exception:
                    pass

                if not error_detail:
                    error_detail = exc.response.text.strip() if exc.response.text else str(exc)

                error_str = f"HTTP {status_code}: {error_detail}"
            else:
                error_str = str(exc)
            # Redact any accidental leakage of the API key in headers or error strings
            if self.api_key and self.api_key in error_str:
                error_str = error_str.replace(self.api_key, "[REDACTED_API_KEY]")

            logger.error(f"Error calling LLM provider '{self.provider}': {error_str}")
            return GeneratedAnswer(
                answer=f"Error generating answer from LLM provider ({self.provider}): {error_str}",
                citations=[],
                sources=sources_summary,
                confidence_score=round(top_score, 4),
                refused=True,
                latency_ms=round(elapsed, 2),
                model=self.model,
                provider=self.provider
            )
