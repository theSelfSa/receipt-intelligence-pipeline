from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.config import Settings
from app.services.matching import ProductMatchingService, _vector_literal


class _FakeMappingsResult:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def mappings(self) -> "_FakeMappingsResult":
        return self

    def all(self) -> list[dict[str, object]]:
        return self._rows

    def first(self):
        return self._rows[0] if self._rows else None


@dataclass
class _FakeProduct:
    id: int
    name: str
    brand: str | None = None
    category: str | None = None


class _FakeSession:
    def __init__(self, rows: list[dict[str, object]], products: dict[int, _FakeProduct]) -> None:
        self._rows = rows
        self._products = products
        self._execute_calls = 0

    async def execute(self, *_args, **_kwargs) -> _FakeMappingsResult:
        self._execute_calls += 1
        if self._execute_calls == 1:
            return _FakeMappingsResult(self._rows)
        return _FakeMappingsResult([])

    async def get(self, _model, product_id: int) -> _FakeProduct | None:
        return self._products.get(product_id)


@pytest.mark.asyncio
async def test_match_line_item_returns_none_for_empty_text() -> None:
    service = ProductMatchingService(Settings())
    session = _FakeSession(rows=[], products={})

    result = await service.match_line_item("   ", session)

    assert result.method == "none"
    assert result.product is None
    assert result.confidence == 0.0


@pytest.mark.asyncio
async def test_match_line_item_uses_fuzzy_match_when_score_is_high() -> None:
    service = ProductMatchingService(Settings(openai_api_key=""))
    session = _FakeSession(
        rows=[{"id": 7, "name": "Organic Whole Milk"}],
        products={7: _FakeProduct(id=7, name="Organic Whole Milk", category="grocery")},
    )

    result = await service.match_line_item("Organic Whole Milk", session)

    assert result.method == "fuzzy"
    assert result.product is not None
    assert result.product.id == 7
    assert result.confidence >= 0.85


def test_vector_literal_serializes_embedding_with_expected_format() -> None:
    values = [0.1, -0.234567891, 1.0]

    literal = _vector_literal(values)

    assert literal == "[0.10000000,-0.23456789,1.00000000]"
