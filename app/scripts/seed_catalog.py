from __future__ import annotations

import argparse
import asyncio
import csv
from pathlib import Path

from sqlalchemy import delete, func, select

import app.database as database
from app.config import get_settings
from app.models.db import CanonicalProductDB


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed canonical product catalog from CSV.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data") / "sample_products.csv",
        help="Path to CSV file with columns: name,brand,category,subcategory,upc",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing products before seeding.",
    )
    return parser.parse_args()


async def seed_catalog(csv_path: Path, reset: bool = False) -> dict[str, int]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows = _read_csv(csv_path)
    settings = get_settings()

    await database.init_database(settings.database_url)
    try:
        if database.session_factory is None:
            raise RuntimeError("Database session factory not initialized.")

        inserted = 0
        updated = 0
        skipped = 0

        async with database.session_factory() as session:
            if reset:
                await session.execute(delete(CanonicalProductDB))
                await session.commit()

            for row in rows:
                normalized = _normalize_row(row)
                if normalized is None:
                    skipped += 1
                    continue

                name, brand, category, subcategory, upc = normalized
                query = select(CanonicalProductDB).where(CanonicalProductDB.name == name)
                if brand is None:
                    query = query.where(CanonicalProductDB.brand.is_(None))
                else:
                    query = query.where(CanonicalProductDB.brand == brand)

                existing = (await session.execute(query.limit(1))).scalars().first()
                if existing is None:
                    session.add(
                        CanonicalProductDB(
                            name=name,
                            brand=brand,
                            category=category,
                            subcategory=subcategory,
                            upc=upc,
                        )
                    )
                    inserted += 1
                else:
                    existing.category = category
                    existing.subcategory = subcategory
                    existing.upc = upc
                    updated += 1

            await session.commit()
            total_count = int(
                (
                    await session.execute(
                        select(func.count()).select_from(CanonicalProductDB)
                    )
                ).scalar()
                or 0
            )
    finally:
        await database.close_database()

    return {
        "inserted": inserted,
        "updated": updated,
        "skipped": skipped,
        "total": total_count,
    }


def _read_csv(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _normalize_row(
    row: dict[str, str | None],
) -> tuple[str, str | None, str, str | None, str | None] | None:
    name = (row.get("name") or "").strip()
    category = (row.get("category") or "").strip()
    if not name or not category:
        return None

    brand = _none_if_blank(row.get("brand"))
    subcategory = _none_if_blank(row.get("subcategory"))
    upc = _none_if_blank(row.get("upc"))
    return name, brand, category, subcategory, upc


def _none_if_blank(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized if normalized else None


def main() -> None:
    args = parse_args()
    result = asyncio.run(seed_catalog(args.csv, reset=args.reset))
    print(
        "Catalog seed complete:"
        f" inserted={result['inserted']},"
        f" updated={result['updated']},"
        f" skipped={result['skipped']},"
        f" total={result['total']}"
    )


if __name__ == "__main__":
    main()
