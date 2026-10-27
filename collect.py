"""Collect one round of ScrapingBee Fast Search results from queries.csv."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
API_URL = "https://app.scrapingbee.com/api/v1/fast_search"
CREDITS_PER_REQUEST = 10
MANIFEST_FIELDS = (
    "run_id",
    "requested_at_utc",
    "query_id",
    "search",
    "country_code",
    "category",
    "http_status",
    "organic_count",
    "response_file",
    "error",
)


@dataclass(frozen=True)
class Query:
    query_id: str
    search: str
    country_code: str
    category: str


def load_queries(path: Path) -> list[Query]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"query_id", "search", "country_code", "category"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV must have columns: {', '.join(sorted(required))}")

        queries: list[Query] = []
        seen_ids: set[str] = set()
        seen_pairs: set[tuple[str, str]] = set()
        for line_no, row in enumerate(reader, start=2):
            query_id = (row.get("query_id") or "").strip()
            search = (row.get("search") or "").strip()
            country = (row.get("country_code") or "").strip().lower()
            category = (row.get("category") or "").strip()
            if not re.fullmatch(r"[a-z0-9_-]+", query_id):
                raise ValueError(f"Line {line_no}: invalid query_id {query_id!r}")
            if not search or not re.fullmatch(r"[a-z]{2}", country):
                raise ValueError(f"Line {line_no}: search or country_code is invalid")
            if query_id in seen_ids or (search.casefold(), country) in seen_pairs:
                raise ValueError(f"Line {line_no}: duplicate query_id or search/country pair")
            seen_ids.add(query_id)
            seen_pairs.add((search.casefold(), country))
            queries.append(Query(query_id, search, country, category))
    if not queries:
        raise ValueError("CSV has no queries")
    return queries


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def collect_once(
    queries: list[Query],
    api_key: str,
    output_root: Path,
    *,
    session: requests.Session,
    delay: float = 0.5,
    run_id: str | None = None,
) -> tuple[Path, int]:
    """Save unmodified API response bodies and one metadata row per request.

    No automatic retries: unexpected repeats can consume additional credits.
    """
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / "manifest.csv"
    failures = 0

    with manifest_path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()

        for index, query in enumerate(queries, start=1):
            requested_at = utc_now()
            print(f"[{index}/{len(queries)}] {query.search!r} ({query.country_code})", flush=True)
            status: int | str = ""
            organic_count: int | str = ""
            response_file = ""
            error = ""
            stop = False
            try:
                response = session.get(
                    API_URL,
                    headers={"Authorization": f"Bearer {api_key}"},
                    params={
                        "search": query.search,
                        "country_code": query.country_code,
                        "page": 1,
                    },
                    timeout=(10, 45),
                )
                status = response.status_code
                if response.ok:
                    # Keep the original response even if it is unexpectedly shaped.
                    path = run_dir / f"{query.query_id}.json"
                    path.write_bytes(response.content)
                    response_file = path.name
                    data: Any = response.json()
                    if not isinstance(data, dict) or not isinstance(data.get("organic"), list):
                        raise ValueError("API response does not contain an organic results list")
                    organic_count = len(data["organic"])
                    print(f"    saved {organic_count} organic results", flush=True)
                else:
                    path = run_dir / f"{query.query_id}.error.txt"
                    path.write_bytes(response.content)
                    response_file = path.name
                    error = f"HTTP {status}"
                    failures += 1
                    print(f"    {error}", file=sys.stderr, flush=True)
                    # Avoid exhausting quota or hammering an invalid credential.
                    stop = status in {401, 402, 403, 429}
            except (requests.RequestException, ValueError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                failures += 1
                print(f"    {error}", file=sys.stderr, flush=True)

            writer.writerow({
                "run_id": run_id,
                "requested_at_utc": requested_at,
                "query_id": query.query_id,
                "search": query.search,
                "country_code": query.country_code,
                "category": query.category,
                "http_status": status,
                "organic_count": organic_count,
                "response_file": response_file,
                "error": error,
            })
            out.flush()
            if stop:
                print("Stopping after authentication, quota, or rate-limit error.", file=sys.stderr)
                break
            if index < len(queries) and delay > 0:
                time.sleep(delay)

    return manifest_path, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=ROOT / "queries.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--limit", type=int, help="Use just the first N rows (for a smoke test)")
    parser.add_argument("--only", metavar="QUERY_ID", help="Run only one query_id")
    parser.add_argument("--delay", type=float, default=0.5, help="Pause between requests, in seconds")
    parser.add_argument("--execute", action="store_true", help="Send paid API requests; otherwise preview")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.delay < 0:
        parser.error("--delay must not be negative")

    try:
        queries = load_queries(args.queries)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.only:
        queries = [q for q in queries if q.query_id == args.only]
        if not queries:
            parser.error(f"Unknown query_id: {args.only}")
    if args.limit is not None:
        queries = queries[:args.limit]

    print(f"Planned: {len(queries)} API calls (~{len(queries) * CREDITS_PER_REQUEST} credits if successful)")
    if not args.execute:
        for query in queries:
            print(f"  {query.query_id:30} {query.country_code}  {query.search}")
        print("Preview only. Pass --execute to request results.")
        return 0

    load_dotenv(ROOT / ".env")
    api_key = os.getenv("SCRAPINGBEE_API_KEY", "").strip()
    if not api_key or api_key == "your_api_key_here":
        parser.error("Set SCRAPINGBEE_API_KEY in .env or as an environment variable")

    with requests.Session() as session:
        manifest, failures = collect_once(
            queries, api_key, args.output, session=session, delay=args.delay
        )
    print(f"Metadata: {manifest}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
