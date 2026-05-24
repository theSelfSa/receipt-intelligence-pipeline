from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from openai import AsyncOpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.models.db import ErrorPatternDB, ReviewQueueDB


class ErrorAnalysisService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: AsyncOpenAI | None = None

    async def analyze_recent_corrections(self, session: AsyncSession, days: int = 7) -> int:
        since = datetime.now(timezone.utc) - timedelta(days=days)
        query = select(ReviewQueueDB).where(
            ReviewQueueDB.status == "corrected",
            ReviewQueueDB.resolved_at.is_not(None),
            ReviewQueueDB.resolved_at >= since,
        )
        result = await session.execute(query)
        corrections = list(result.scalars().all())
        if len(corrections) < 5:
            return 0

        grouped: dict[str, list[ReviewQueueDB]] = defaultdict(list)
        for correction in corrections:
            key = correction.error_type or "unknown"
            grouped[key].append(correction)

        suggestions = await self._build_suggestions(grouped)
        inserted = 0

        for error_type, group in grouped.items():
            suggestion = suggestions.get(error_type, {})
            pattern = ErrorPatternDB(
                error_type=error_type,
                description=suggestion.get("description") or _default_description(error_type, group),
                example_receipt_ids=_example_receipt_ids(group),
                occurrence_count=len(group),
                suggested_prompt_fix=suggestion.get("suggested_prompt_fix") or _default_prompt_fix(error_type),
            )
            session.add(pattern)
            inserted += 1

        await session.commit()
        return inserted

    async def _build_suggestions(
        self,
        grouped: dict[str, list[ReviewQueueDB]],
    ) -> dict[str, dict[str, str]]:
        if not self._settings.openai_api_key:
            return {}

        payload = {
            error_type: [
                {
                    "field_name": item.field_name,
                    "extracted_value": item.extracted_value,
                    "corrected_value": item.corrected_value,
                    "reviewer_notes": item.reviewer_notes,
                }
                for item in items[:8]
            ]
            for error_type, items in grouped.items()
        }

        prompt = (
            "Given corrected receipt extraction examples grouped by error type, return JSON object where each key "
            "is error type and value has keys: description, suggested_prompt_fix. Keep each suggestion short."
        )

        try:
            client = self._get_client()
            response = await client.chat.completions.create(
                model=self._settings.openai_model,
                messages=[
                    {"role": "system", "content": "You are an expert in OCR + LLM extraction error analysis."},
                    {
                        "role": "user",
                        "content": f"{prompt}\n\nInput JSON:\n{json.dumps(payload)}",
                    },
                ],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content if response.choices else None
            if not content:
                return {}
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                normalized: dict[str, dict[str, str]] = {}
                for key, value in parsed.items():
                    if isinstance(value, dict):
                        normalized[key] = {
                            "description": str(value.get("description") or ""),
                            "suggested_prompt_fix": str(value.get("suggested_prompt_fix") or ""),
                        }
                return normalized
        except Exception:
            return {}

        return {}

    def _get_client(self) -> AsyncOpenAI:
        if self._client is not None:
            return self._client
        self._client = AsyncOpenAI(api_key=self._settings.openai_api_key)
        return self._client


def _example_receipt_ids(items: list[ReviewQueueDB]) -> list[int]:
    ids: list[int] = []
    for item in items:
        if item.receipt_id is None:
            continue
        ids.append(item.receipt_id)
    return ids[:5]


def _default_description(error_type: str, items: list[ReviewQueueDB]) -> str:
    field_names = sorted({item.field_name for item in items if item.field_name})
    fields_text = ", ".join(field_names[:3]) if field_names else "multiple fields"
    return f"Frequent {error_type} corrections detected across {fields_text}."


def _default_prompt_fix(error_type: str) -> str:
    defaults: dict[str, str] = {
        "date_format": "Normalize dates to ISO format and prefer explicit month/day/year parsing.",
        "quantity_parse": "Validate quantity patterns (QTY, EA, LB, OZ) and avoid defaulting to 1 without evidence.",
        "price_mismatch": "Cross-check line-item totals against subtotal and surface mismatches as warnings.",
        "category_wrong": "Infer category from both merchant context and line-item vocabulary before assigning.",
        "unknown": "When uncertain, return null and provide conservative confidence scores.",
    }
    return defaults.get(
        error_type,
        "Add explicit extraction constraints for this error type and return null when uncertain.",
    )
