"""Compare Ollama relevance reports and organic SERP hosts across collection runs.

Reads results/<run_id>/analysis/{summary.json,analysis.csv}; never calls an API.
Synthetic controls are excluded unless --include-synthetic is specified.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
LABELS = ("relevant", "partial", "irrelevant", "uncertain")
QUERY_FIELDS: tuple[str, ...] = (
    "query_id", "search", "country_code", "previous_run", "previous_utc",
    "current_run", "current_utc", "previous_hosts", "current_hosts",
    "shared_hosts", "previous_retention_pct", "current_retention_pct",
    "jaccard_pct", "shared_hosts_with_rank_changes", "mean_absolute_rank_change",
    "previous_relevant", "previous_partial", "previous_irrelevant", "previous_uncertain",
    "current_relevant", "current_partial", "current_irrelevant", "current_uncertain",
    "new_hosts", "lost_hosts", "review_flags",
)
QUERY_RUN_FIELDS: tuple[str, ...] = (
    "query_id", "search", "country_code", "run_id", "requested_at_utc",
    "model", "prompt_version", "results", "unique_hosts",
    "relevant", "partial", "irrelevant", "uncertain",
    "topically_related_pct", "irrelevant_pct", "uncertain_pct",
)
AGGREGATE_FIELDS: tuple[str, ...] = (
    "query_id", "search", "country_code", "runs", "first_utc", "last_utc",
    "results_total", "relevant", "partial", "irrelevant", "uncertain",
    "topically_related_pct", "irrelevant_pct", "uncertain_pct",
    "runs_with_irrelevant", "runs_with_uncertain",
    "unique_hosts_across_runs", "hosts_in_every_run",
    "comparisons", "average_previous_retention_pct", "min_previous_retention_pct",
    "average_jaccard_pct", "min_jaccard_pct", "low_overlap_pairs",
    "flagged_pairs", "new_host_events", "lost_host_events",
)
RUN_FIELDS: tuple[str, ...] = (
    "run_id", "first_request_utc", "model", "prompt_version", "queries_analyzed",
    "queries_skipped", "queries_failed", "queries_empty", "results_analyzed",
    "relevant", "partial", "irrelevant", "uncertain", "topically_related_pct",
    "irrelevant_pct",
)


@dataclass(frozen=True)
class Serp:
    run_id: str
    query_id: str
    search: str
    country: str
    when: datetime
    when_text: str
    model: str
    prompt_version: str
    counts: dict[str, int]
    # A host may have multiple results; retain its BEST (smallest) position.
    host_ranks: dict[str, int]


def parse_time(value: str, context: str) -> datetime:
    if not value:
        raise ValueError(f"{context}: missing requested_at_utc")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{context}: invalid timestamp {value!r}") from exc
    if result.tzinfo is None:
        raise ValueError(f"{context}: timestamp must include timezone")
    return result.astimezone(timezone.utc)


def host_from_result(row: dict[str, str]) -> str:
    # `domain` in analyze.py is a hostname with leading www. removed. It is
    # NOT a registered/organizational domain (e.g., docs.example.com differs
    # from example.com). Avoid an unreliable hand-written public-suffix parser.
    domain = (row.get("domain") or "").strip().lower().removeprefix("www.")
    if domain:
        return domain
    return (urlsplit(row.get("link") or "").hostname or "").lower().removeprefix("www.")


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or ()))
            raise ValueError(f"{path}: missing CSV columns: {', '.join(missing)}")
        return list(reader)


def is_synthetic(run_dir: Path) -> bool:
    if run_dir.name.startswith("synthetic_"):
        return True
    manifest = run_dir / "manifest.csv"
    if manifest.is_file():
        rows = read_csv(manifest, {"category"})
        return any((r.get("category") or "").lower().startswith("synthetic") for r in rows)
    return False


def load_run(run_dir: Path) -> tuple[dict[str, Any], list[Serp]]:
    analysis = run_dir / "analysis"
    summary_path = analysis / "summary.json"
    csv_path = analysis / "analysis.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    if not isinstance(summary, dict) or not isinstance(summary.get("per_query"), list):
        raise ValueError(f"{summary_path}: invalid report")
    run_id = summary.get("run_id")
    if not isinstance(run_id, str) or run_id != run_dir.name:
        raise ValueError(f"{summary_path}: run_id does not match folder name")
    model = summary.get("model")
    prompt_version = summary.get("prompt_version")
    if not isinstance(model, str) or not isinstance(prompt_version, str):
        raise ValueError(f"{summary_path}: model/prompt_version missing")
    rows = read_csv(csv_path, {"run_id", "query_id", "search", "country_code", "rank", "relevance", "domain", "link"})
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["run_id"] != run_id:
            raise ValueError(f"{csv_path}: row run_id mismatch")
        if row["relevance"] not in LABELS:
            raise ValueError(f"{csv_path}: unexpected relevance label {row['relevance']!r}")
        grouped[row["query_id"]].append(row)

    seen_ids: set[str] = set()
    serps: list[Serp] = []
    for query in summary["per_query"]:
        if not isinstance(query, dict):
            raise ValueError(f"{summary_path}: invalid per_query entry")
        query_id = query.get("query_id")
        if not isinstance(query_id, str) or not query_id or query_id in seen_ids:
            raise ValueError(f"{summary_path}: invalid/duplicate query_id {query_id!r}")
        seen_ids.add(query_id)
        status = query.get("status")
        entry_rows = grouped.get(query_id, [])
        if status != "analyzed":
            if entry_rows:
                raise ValueError(f"{csv_path}: rows present for non-analyzed query {query_id}")
            continue
        search = query.get("search")
        country = query.get("country_code")
        when_text = query.get("requested_at_utc")
        if not isinstance(search, str) or not isinstance(country, str) or not isinstance(when_text, str):
            raise ValueError(f"{summary_path}: missing query metadata for {query_id}")
        when = parse_time(when_text, f"{run_id}/{query_id}")
        organic_count = query.get("organic_count")
        if len(entry_rows) != organic_count or not entry_rows:
            raise ValueError(f"{run_id}/{query_id}: report row count differs from organic_count")

        counts: Counter[str] = Counter()
        host_ranks: dict[str, int] = {}
        seen_ranks: set[int] = set()
        for row in entry_rows:
            if row["search"] != search or row["country_code"].lower() != country.lower():
                raise ValueError(f"{csv_path}: query metadata mismatch for {query_id}")
            try:
                rank = int(row["rank"])
            except ValueError as exc:
                raise ValueError(f"{csv_path}: invalid rank {row['rank']!r}") from exc
            if rank < 1 or rank in seen_ranks:
                raise ValueError(f"{csv_path}: duplicated/invalid rank for {query_id}")
            seen_ranks.add(rank)
            host = host_from_result(row)
            if not host:
                raise ValueError(f"{csv_path}: missing hostname at {query_id} rank {rank}")
            host_ranks[host] = min(rank, host_ranks.get(host, rank))
            counts[row["relevance"]] += 1
        expected_counts = query.get("counts")
        if not isinstance(expected_counts, dict) or any(counts[label] != expected_counts.get(label) for label in LABELS):
            raise ValueError(f"{run_id}/{query_id}: relevance counts differ from summary.json")
        serps.append(Serp(
            run_id, query_id, search, country.lower(), when, when_text,
            model, prompt_version, {label: counts[label] for label in LABELS}, host_ranks,
        ))

    if set(grouped) - seen_ids:
        raise ValueError(f"{csv_path}: rows for query_ids absent from summary.json")
    return summary, serps


def compare_pair(previous: Serp, current: Serp, alert_overlap: float) -> tuple[dict[str, Any], dict[str, Any]]:
    old = set(previous.host_ranks)
    new = set(current.host_ranks)
    shared = old & new
    lost = sorted(old - new)
    added = sorted(new - old)
    prev_ret = len(shared) / len(old) * 100
    curr_ret = len(shared) / len(new) * 100
    jaccard = len(shared) / len(old | new) * 100
    shifts = {host: current.host_ranks[host] - previous.host_ranks[host] for host in sorted(shared)}
    flags: list[str] = []
    if prev_ret < alert_overlap * 100:
        flags.append("low_host_overlap")
    if current.counts["irrelevant"]:
        flags.append("irrelevant_results_present")
    if current.counts["uncertain"]:
        flags.append("uncertain_results_present")
    record: dict[str, Any] = {
        "query_id": current.query_id,
        "search": current.search,
        "country_code": current.country,
        "previous_run": previous.run_id,
        "previous_utc": previous.when_text,
        "current_run": current.run_id,
        "current_utc": current.when_text,
        "previous_hosts": len(old),
        "current_hosts": len(new),
        "shared_hosts": len(shared),
        "previous_retention_pct": round(prev_ret, 2),
        "current_retention_pct": round(curr_ret, 2),
        "jaccard_pct": round(jaccard, 2),
        "shared_hosts_with_rank_changes": sum(delta != 0 for delta in shifts.values()),
        "mean_absolute_rank_change": round(sum(abs(delta) for delta in shifts.values()) / len(shifts), 2) if shifts else "",
        "new_hosts": "; ".join(added),
        "lost_hosts": "; ".join(lost),
        "review_flags": "; ".join(flags),
    }
    for label in LABELS:
        record[f"previous_{label}"] = previous.counts[label]
        record[f"current_{label}"] = current.counts[label]
    detail = {"query_id": current.query_id, "previous_run": previous.run_id,
              "current_run": current.run_id, "shared_rank_changes": shifts,
              "new_hosts": added, "lost_hosts": lost, "review_flags": flags}
    return record, detail


def run_row(summary: dict[str, Any], serps: list[Serp]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for serp in serps:
        counts.update(serp.counts)
    total = sum(counts.values())
    return {
        "run_id": summary["run_id"],
        "first_request_utc": min((s.when for s in serps)).isoformat() if serps else "",
        "model": summary["model"],
        "prompt_version": summary["prompt_version"],
        "queries_analyzed": len(serps),
        "queries_skipped": summary.get("queries_skipped", 0),
        "queries_failed": summary.get("queries_failed", 0),
        "queries_empty": summary.get("queries_empty", 0),
        "results_analyzed": total,
        **{label: counts[label] for label in LABELS},
        "topically_related_pct": round(100 * (counts["relevant"] + counts["partial"]) / total, 2) if total else "",
        "irrelevant_pct": round(100 * counts["irrelevant"] / total, 2) if total else "",
    }


def aggregate_by_query(
    by_query: dict[str, list[Serp]], comparisons: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build a longitudinal dataset without modifying individual run reports.

    Percentages for relevance use pooled RESULT counts. Overlap averages use
    one value per consecutive SERP comparison. A single-run query has no
    overlap estimate (blank CSV cells, not zero).
    """
    pair_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for comparison in comparisons:
        pair_rows[comparison["query_id"]].append(comparison)

    query_runs: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for query_id, serps in sorted(by_query.items()):
        # build_report() sorts these already; sort again for standalone safety.
        ordered = sorted(serps, key=lambda s: (s.when, s.run_id))
        totals: Counter[str] = Counter()
        all_hosts: set[str] = set()
        common_hosts: set[str] | None = None
        for serp in ordered:
            counts = serp.counts
            totals.update(counts)
            hosts = set(serp.host_ranks)
            all_hosts.update(hosts)
            common_hosts = hosts if common_hosts is None else common_hosts & hosts
            result_count = sum(counts.values())
            query_runs.append({
                "query_id": query_id,
                "search": serp.search,
                "country_code": serp.country,
                "run_id": serp.run_id,
                "requested_at_utc": serp.when_text,
                "model": serp.model,
                "prompt_version": serp.prompt_version,
                "results": result_count,
                "unique_hosts": len(hosts),
                **{label: counts[label] for label in LABELS},
                "topically_related_pct": round(100 * (counts["relevant"] + counts["partial"]) / result_count, 2),
                "irrelevant_pct": round(100 * counts["irrelevant"] / result_count, 2),
                "uncertain_pct": round(100 * counts["uncertain"] / result_count, 2),
            })

        pairs = pair_rows[query_id]
        retention = [float(pair["previous_retention_pct"]) for pair in pairs]
        jaccard = [float(pair["jaccard_pct"]) for pair in pairs]
        total = sum(totals.values())
        summaries.append({
            "query_id": query_id,
            "search": ordered[0].search,
            "country_code": ordered[0].country,
            "runs": len(ordered),
            "first_utc": ordered[0].when_text,
            "last_utc": ordered[-1].when_text,
            "results_total": total,
            **{label: totals[label] for label in LABELS},
            "topically_related_pct": round(100 * (totals["relevant"] + totals["partial"]) / total, 2),
            "irrelevant_pct": round(100 * totals["irrelevant"] / total, 2),
            "uncertain_pct": round(100 * totals["uncertain"] / total, 2),
            "runs_with_irrelevant": sum(s.counts["irrelevant"] > 0 for s in ordered),
            "runs_with_uncertain": sum(s.counts["uncertain"] > 0 for s in ordered),
            "unique_hosts_across_runs": len(all_hosts),
            "hosts_in_every_run": len(common_hosts or set()),
            "comparisons": len(pairs),
            "average_previous_retention_pct": round(mean(retention), 2) if retention else "",
            "min_previous_retention_pct": round(min(retention), 2) if retention else "",
            "average_jaccard_pct": round(mean(jaccard), 2) if jaccard else "",
            "min_jaccard_pct": round(min(jaccard), 2) if jaccard else "",
            "low_overlap_pairs": sum("low_host_overlap" in pair["review_flags"].split("; ") for pair in pairs),
            "flagged_pairs": sum(bool(pair["review_flags"]) for pair in pairs),
            "new_host_events": sum(len(pair["new_hosts"].split("; ")) for pair in pairs if pair["new_hosts"]),
            "lost_host_events": sum(len(pair["lost_hosts"].split("; ")) for pair in pairs if pair["lost_hosts"]),
        })

    query_runs.sort(key=lambda row: (parse_time(row["requested_at_utc"], row["query_id"]), row["query_id"]))
    return query_runs, summaries


def build_report(
    results_dir: Path, *, include_synthetic: bool = False,
    allow_mixed_evaluators: bool = False, query_id: str | None = None,
    alert_overlap: float = 0.5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not results_dir.is_dir():
        raise ValueError(f"Results directory not found: {results_dir}")
    if not 0 <= alert_overlap <= 1:
        raise ValueError("--alert-overlap must be between 0 and 1")

    by_query: dict[str, list[Serp]] = defaultdict(list)
    run_summaries: list[dict[str, Any]] = []
    skipped_dirs: list[dict[str, str]] = []
    identities: dict[str, tuple[str, str]] = {}
    evaluators: set[tuple[str, str]] = set()
    known_run_ids: set[str] = set()
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir() or not (run_dir / "manifest.csv").is_file():
            continue  # e.g. results/comparison is not a collection run
        if not include_synthetic and is_synthetic(run_dir):
            skipped_dirs.append({"run_dir": run_dir.name, "reason": "synthetic test excluded"})
            continue
        if not (run_dir / "analysis" / "summary.json").is_file() or not (run_dir / "analysis" / "analysis.csv").is_file():
            skipped_dirs.append({"run_dir": run_dir.name, "reason": "analysis report missing"})
            continue
        summary, serps = load_run(run_dir)
        if summary["run_id"] in known_run_ids:
            raise ValueError(f"Duplicate run_id {summary['run_id']}")
        known_run_ids.add(summary["run_id"])
        if query_id is not None:
            serps = [s for s in serps if s.query_id == query_id]
        if not serps:
            skipped_dirs.append({"run_dir": run_dir.name, "reason": "no matching analyzed queries"})
            continue
        evaluators.add((summary["model"], summary["prompt_version"]))
        for serp in serps:
            identity = (serp.search.casefold().strip(), serp.country)
            previous_identity = identities.setdefault(serp.query_id, identity)
            if identity != previous_identity:
                raise ValueError(f"query_id {serp.query_id!r} changed search/country across runs")
            by_query[serp.query_id].append(serp)
        run_summaries.append(run_row(summary, serps))

    if len(evaluators) > 1 and not allow_mixed_evaluators:
        labels = ", ".join(f"{m} / {p}" for m, p in sorted(evaluators))
        raise ValueError(f"Different model/prompt versions: {labels}. Analyze with one evaluator or pass --allow-mixed-evaluators.")

    comparisons: list[dict[str, Any]] = []
    comparison_details: list[dict[str, Any]] = []
    for _, serps in sorted(by_query.items()):
        serps.sort(key=lambda serp: (serp.when, serp.run_id))
        for previous, current in zip(serps, serps[1:]):
            row, details = compare_pair(previous, current, alert_overlap)
            comparisons.append(row)
            comparison_details.append(details)
    comparisons.sort(key=lambda row: (row["current_utc"], row["query_id"]))
    run_summaries.sort(key=lambda row: (row["first_request_utc"], row["run_id"]))
    all_counts: Counter[str] = Counter()
    for row in run_summaries:
        for label in LABELS:
            all_counts[label] += int(row[label])
    total = sum(all_counts.values())
    query_runs, query_summaries = aggregate_by_query(by_query, comparisons)
    overview: dict[str, Any] = {
        "runs_included": len(run_summaries),
        "queries_seen": len(by_query),
        "consecutive_pairs_compared": len(comparisons),
        "model_prompt_pairs": [{"model": model, "prompt_version": prompt} for model, prompt in sorted(evaluators)],
        "aggregate_results": total,
        "aggregate_counts": {label: all_counts[label] for label in LABELS},
        "topically_related_pct": round((all_counts["relevant"] + all_counts["partial"]) * 100 / total, 2) if total else None,
        "irrelevant_pct": round(all_counts["irrelevant"] * 100 / total, 2) if total else None,
        "flagged_comparisons": sum(bool(row["review_flags"]) for row in comparisons),
        "skipped_directories": skipped_dirs,
        "comparison_details": comparison_details,
        "query_runs": query_runs,
        "query_summaries": query_summaries,
        "notes": [
            "Retention = shared unique hostnames / previous run unique hostnames; Jaccard = shared / union.",
            "Hostnames are not registrable domains: blog.example.com differs from example.com.",
            "Only consecutive available analyzed runs for the same query_id are compared; intervening failed/missing runs may exist.",
            "Relevance uses Ollama judgments of snippets, not target page content; flags are manual-review prompts, NOT proof of poisoning.",
            "Synthetic control runs are excluded by default; relevance counts include only analyzed SERPs.",
            "Query summary relevance percentages are pooled across results; overlap averages are means of consecutive-pair percentages.",
            "Single-run queries have no overlap statistics; blank cells mean insufficient data, not zero.",
            "Local 'near me' queries are evaluated topically without a verified city; country-only location is not a proximity test.",
        ],
    }
    return run_summaries, comparisons, overview


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results", help="Parent results folder")
    parser.add_argument("--output", type=Path, help="Output folder (default: RESULTS/comparison)")
    parser.add_argument("--query-id", help="Limit comparisons to one query_id")
    parser.add_argument("--alert-overlap", type=float, default=0.5, help="Flag previous-run hostname retention below this fraction (default: 0.5)")
    parser.add_argument("--include-synthetic", action="store_true", help="Include synthetic control data (normally excluded)")
    parser.add_argument("--allow-mixed-evaluators", action="store_true", help="Permit different LLM models/prompt versions (not recommended for relevance comparisons)")
    args = parser.parse_args()
    output_dir = args.output or args.results / "comparison"
    try:
        run_rows, comparison_rows, overview = build_report(
            args.results, include_synthetic=args.include_synthetic,
            allow_mixed_evaluators=args.allow_mixed_evaluators,
            query_id=args.query_id, alert_overlap=args.alert_overlap,
        )
        if not run_rows:
            raise ValueError("No analyzed collection runs found (check missing reports or filters)")
        output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(output_dir / "runs.csv", RUN_FIELDS, run_rows)
        write_csv(output_dir / "comparisons.csv", QUERY_FIELDS, comparison_rows)
        write_csv(output_dir / "query_runs.csv", QUERY_RUN_FIELDS, overview["query_runs"])
        write_csv(output_dir / "query_summary.csv", AGGREGATE_FIELDS, overview["query_summaries"])
        (output_dir / "overview.json").write_text(
            json.dumps(overview, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"Included {overview['runs_included']} runs, {overview['queries_seen']} queries, {overview['consecutive_pairs_compared']} comparisons")
    print(f"Flagged for review: {overview['flagged_comparisons']}")
    for item in overview["comparison_details"]:
        if item["review_flags"]:
            print(
                f"  {item['query_id']}: "
                f"{item['previous_run']} -> {item['current_run']} "
                f"[{', '.join(item['review_flags'])}]"
            )
    print(f"Aggregated query summaries: {len(overview['query_summaries'])}")
    print(f"Reports: {output_dir.resolve()}")
    if not comparison_rows:
        print("No comparable pairs yet. Need 2 analyzed runs for the same query_id.")
    if overview["skipped_directories"]:
        print("Skipped:")
        for skipped in overview["skipped_directories"]:
            print(f"  {skipped['run_dir']}: {skipped['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
