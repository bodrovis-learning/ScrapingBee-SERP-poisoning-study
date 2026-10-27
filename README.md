# ScrapingBee SERP reliability study

Collect repeatable organic Google search results using [ScrapingBee Fast Search API](https://www.scrapingbee.com/documentation/fast-search/) to investigate reports of anomalous SERPs.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- A ScrapingBee API key and available credits

## Setup

```bash
uv sync
```

Copy `.env.example` to `.env` and replace the placeholder with your ScrapingBee API key. On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

`.env` is git-ignored. You can also provide `SCRAPINGBEE_API_KEY` as an environment variable instead.

## Queries

Edit `queries.csv` to add or remove query/country pairs. Columns:

- `query_id`: unique filename-safe identifier for each pair
- `search`: the exact Google query
- `country_code`: two-letter country code (`us`, `ca`, `gb`, etc.)
- `category`: internal research label; **not** an API parameter

There are 8 query/country pairs in the initial dataset. The `near me` cases use country-only localization; they **do not** reproduce a precise city or ZIP/postcode location.

## Collect

```bash
# Show the planned requests, without spending credits:
uv run python collect.py

# Make ONE real request first:
uv run python collect.py --execute --limit 1

# Run all 8 requests in one collection round:
uv run python collect.py --execute

# Or choose a single query:
uv run python collect.py --execute --only gold_buyers_ca
```

Each successful Fast Search request costs **10 credits** (according to the documentation). One full round costs approximately **80 credits**. Two rounds daily for 7 days = **112 requests / 1,120 credits**, assuming successful requests and no extra reruns.

**The script does not automatically retry failed requests**, to prevent surprise charges. It stops a round on HTTP 401/402/403/429. There is a 0.5-second pause between requests (adjustable with `--delay`).

For the planned experiment, invoke `uv run python collect.py --execute` **twice daily** about 12 hours apart, using a local scheduler if desired. The script runs once and exits; it does not schedule itself.

## Output

Each round creates a unique timestamped directory:

```text
results/
  20261020T154512_123456Z/
    manifest.csv
    white_label_local_seo_us.json
    white_label_local_seo_ca.json
    ...
```

- `<query_id>.json` contains the original JSON body returned by ScrapingBee, including the `organic` list (`rank`, `title`, `link`, `description`, etc.).
- `manifest.csv` records each request's UTC timestamp, query, country, HTTP status, result count, output file, and any error.
- HTTP failures are saved as `<query_id>.error.txt` for debugging. Network failures are recorded in the manifest.

The script deliberately **does not classify or filter results**. Keep the raw data for later comparison of domains, ranks, relevance, and recovery across runs. A successful API response alone does not prove that a SERP is correct.

`results/` is git-ignored; preserve or back up it between runs. No API calls are made until `--execute` is passed.
