from typing import Any

import instructor
from openai import AsyncOpenAI

from app.models.schemas import Receipt


SYSTEM_PROMPT = """
You are a receipt parsing expert. Extract ALL information from the receipt text.
Be precise with numbers — do not round or estimate prices.
If a field is unclear or missing, set it to null rather than guessing.
For each line item, include your confidence (0-1) that the extraction is correct.
Common abbreviations: QTY=quantity, EA=each, LB=pound, OZ=ounce, TAX=tax, SUBTL=subtotal.
""".strip()


class ReceiptExtractionService:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        client: Any | None = None,
        base_url: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._client = client
        self._base_url = base_url

    async def extract(self, raw_text: str) -> Receipt:
        if not raw_text.strip():
            raise ValueError("OCR text is empty.")

        client = self._get_or_init_client()
        response = await client.chat.completions.create(
            model=self._model,
            response_model=Receipt,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Parse this receipt:\n\n{raw_text}"},
            ],
            max_retries=3,
        )

        if isinstance(response, Receipt):
            return response
        return Receipt.model_validate(response)

    def _get_or_init_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY (or provider API key) is required for structured extraction.")
        client_kwargs: dict[str, Any] = {"api_key": self._api_key}
        if self._base_url:
            client_kwargs["base_url"] = self._base_url
        self._client = instructor.from_openai(AsyncOpenAI(**client_kwargs))
        return self._client
