from dataclasses import dataclass

from openai import AsyncOpenAI
from rapidfuzz import fuzz, process
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.models.db import CanonicalProductDB


@dataclass
class MatchResult:
    method: str
    product: CanonicalProductDB | None
    confidence: float


class ProductMatchingService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._embedding_client: AsyncOpenAI | None = None

    async def match_line_item(self, raw_text: str, session: AsyncSession) -> MatchResult:
        raw_text = raw_text.strip()
        if not raw_text:
            return MatchResult(method="none", product=None, confidence=0.0)

        products_result = await session.execute(text("SELECT id, name FROM canonical_products"))
        products = products_result.mappings().all()
        if not products:
            return MatchResult(method="none", product=None, confidence=0.0)

        choices = {int(row["id"]): str(row["name"]) for row in products}
        fuzzy = process.extractOne(raw_text, choices, scorer=fuzz.token_sort_ratio, score_cutoff=75)
        if fuzzy:
            _, score, product_id = fuzzy
            if score >= 85:
                product = await session.get(CanonicalProductDB, int(product_id))
                if product is not None:
                    return MatchResult(method="fuzzy", product=product, confidence=float(score) / 100.0)

        if not self._settings.openai_api_key:
            return MatchResult(method="none", product=None, confidence=0.0)

        embedding = await self._embed_text(raw_text)
        embedding_literal = _vector_literal(embedding)
        vector_query = text(
            """
            SELECT id, 1 - (embedding <=> CAST(:embedding AS vector)) AS similarity
            FROM canonical_products
            WHERE embedding IS NOT NULL
              AND 1 - (embedding <=> CAST(:embedding AS vector)) > :threshold
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT 1
            """
        )
        vector_result = await session.execute(
            vector_query,
            {"embedding": embedding_literal, "threshold": 0.80},
        )
        best = vector_result.mappings().first()
        if best:
            product = await session.get(CanonicalProductDB, int(best["id"]))
            if product is not None:
                return MatchResult(
                    method="embedding",
                    product=product,
                    confidence=float(best["similarity"]),
                )

        return MatchResult(method="none", product=None, confidence=0.0)

    async def recompute_embeddings(self, session: AsyncSession, force: bool = False) -> int:
        if not self._settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for embedding operations.")

        if force:
            query = text("SELECT id, name, brand, category FROM canonical_products ORDER BY id")
            products_result = await session.execute(query)
        else:
            query = text(
                """
                SELECT id, name, brand, category
                FROM canonical_products
                WHERE embedding IS NULL
                ORDER BY id
                """
            )
            products_result = await session.execute(query)

        rows = products_result.mappings().all()
        if not rows:
            return 0

        ids = [int(row["id"]) for row in rows]
        inputs = [
            f"{row['name']} {row['brand'] or ''} {row['category'] or ''}".strip()
            for row in rows
        ]

        client = self._get_embedding_client()
        response = await client.embeddings.create(
            model=self._settings.openai_embedding_model,
            input=inputs,
        )

        embeddings_by_id = {
            ids[idx]: response.data[idx].embedding
            for idx in range(len(ids))
        }

        updated = 0
        for product_id, embedding in embeddings_by_id.items():
            product = await session.get(CanonicalProductDB, product_id)
            if product is None:
                continue
            product.embedding = embedding
            updated += 1

        await session.commit()
        return updated

    async def _embed_text(self, raw_text: str) -> list[float]:
        client = self._get_embedding_client()
        response = await client.embeddings.create(
            model=self._settings.openai_embedding_model,
            input=raw_text,
        )
        return response.data[0].embedding

    def _get_embedding_client(self) -> AsyncOpenAI:
        if self._embedding_client is not None:
            return self._embedding_client
        if not self._settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for embedding operations.")
        self._embedding_client = AsyncOpenAI(api_key=self._settings.openai_api_key)
        return self._embedding_client


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"
