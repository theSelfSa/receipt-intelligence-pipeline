from fastapi import APIRouter, Depends, Query, status
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.database import get_session
from app.models.api import (
    CatalogEmbedResponse,
    CatalogMatchRequest,
    CatalogMatchResponse,
    CatalogMatchResult,
    CatalogProductCreateRequest,
    CatalogProductResponse,
    CatalogProductsListResponse,
)
from app.models.db import CanonicalProductDB
from app.services.matching import ProductMatchingService

router = APIRouter(prefix="/catalog", tags=["catalog"])


@router.get("/products", response_model=CatalogProductsListResponse)
async def list_products(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
) -> CatalogProductsListResponse:
    query = (
        select(CanonicalProductDB)
        .order_by(CanonicalProductDB.id.asc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    count_query = select(func.count()).select_from(CanonicalProductDB)

    rows = (await session.execute(query)).scalars().all()
    total = int((await session.execute(count_query)).scalar() or 0)
    items = [_to_catalog_product_response(row) for row in rows]
    return CatalogProductsListResponse(items=items, page=page, page_size=page_size, total=total)


@router.post("/products", response_model=CatalogProductResponse, status_code=status.HTTP_201_CREATED)
async def create_product(
    payload: CatalogProductCreateRequest,
    session: AsyncSession = Depends(get_session),
) -> CatalogProductResponse:
    product = CanonicalProductDB(
        name=payload.name,
        brand=payload.brand,
        category=payload.category,
        subcategory=payload.subcategory,
        upc=payload.upc,
    )
    session.add(product)
    await session.commit()
    await session.refresh(product)
    return _to_catalog_product_response(product)


@router.get("/search", response_model=CatalogProductsListResponse)
async def search_products(
    q: str = Query(..., min_length=1),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
) -> CatalogProductsListResponse:
    filter_clause = or_(
        CanonicalProductDB.name.ilike(f"%{q}%"),
        CanonicalProductDB.brand.ilike(f"%{q}%"),
    )
    query = (
        select(CanonicalProductDB)
        .where(filter_clause)
        .order_by(CanonicalProductDB.id.asc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    count_query = select(func.count()).select_from(CanonicalProductDB).where(filter_clause)

    rows = (await session.execute(query)).scalars().all()
    total = int((await session.execute(count_query)).scalar() or 0)
    items = [_to_catalog_product_response(row) for row in rows]
    return CatalogProductsListResponse(items=items, page=page, page_size=page_size, total=total)


@router.post("/match", response_model=CatalogMatchResponse)
async def match_product(
    payload: CatalogMatchRequest,
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> CatalogMatchResponse:
    matcher = ProductMatchingService(settings)
    result = await matcher.match_line_item(payload.raw_text, session)

    return CatalogMatchResponse(
        query=payload.raw_text,
        result=CatalogMatchResult(
            method=result.method,
            product_id=result.product.id if result.product is not None else None,
            name=result.product.name if result.product is not None else None,
            brand=result.product.brand if result.product is not None else None,
            category=result.product.category if result.product is not None else None,
            confidence=result.confidence,
        ),
    )


@router.post("/embed", response_model=CatalogEmbedResponse)
async def embed_products(
    force: bool = Query(default=False),
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> CatalogEmbedResponse:
    matcher = ProductMatchingService(settings)
    updated = await matcher.recompute_embeddings(session, force=force)
    return CatalogEmbedResponse(updated_count=updated, model=settings.openai_embedding_model)


def _to_catalog_product_response(product: CanonicalProductDB) -> CatalogProductResponse:
    return CatalogProductResponse(
        id=product.id,
        name=product.name,
        brand=product.brand,
        category=product.category,
        subcategory=product.subcategory,
        upc=product.upc,
        created_at=product.created_at,
        updated_at=product.updated_at,
    )
