"""Download 10-K annual reports from SEC EDGAR and convert them to text.

Why 10-Ks: they are public domain, enormous, and pathologically repetitive --
the same risk-factor and accounting-policy boilerplate recurs within a filing and
across filings from different companies. That cross-document redundancy is
exactly what stage [4] (MMR) exists to kill, so it makes the demo honest rather
than flattering.

Table handling matters. The gold set needs "number buried in a table" questions,
and stage [2] must treat tables as atomic units. So tables are converted to
pipe-delimited lines wrapped in explicit markers that the sentence splitter can
detect, instead of being flattened into unparseable number soup.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import httpx

from ..config import CORPUS_DIR, settings

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"

# Different sectors on purpose: the boilerplate overlaps heavily across them.
DEFAULT_TICKERS = ("AAPL", "MSFT", "JPM", "JNJ", "KO")

TABLE_OPEN = "[TABLE]"
TABLE_CLOSE = "[/TABLE]"


def _client() -> httpx.Client:
    # EDGAR rejects requests without a contact string in the User-Agent.
    return httpx.Client(
        headers={"User-Agent": settings.sec_user_agent, "Accept-Encoding": "gzip, deflate"},
        timeout=60.0,
        follow_redirects=True,
    )


def _resolve_ciks(client: httpx.Client, tickers: tuple[str, ...]) -> dict[str, int]:
    raw = client.get(TICKER_MAP_URL).json()
    by_ticker = {row["ticker"].upper(): int(row["cik_str"]) for row in raw.values()}
    missing = [t for t in tickers if t.upper() not in by_ticker]
    if missing:
        raise RuntimeError(f"Tickers not found in EDGAR map: {missing}")
    return {t.upper(): by_ticker[t.upper()] for t in tickers}


def _latest_10k(client: httpx.Client, cik: int) -> tuple[str, str, str]:
    """Return (accession_no_nodashes, primary_document, filing_date)."""
    data = client.get(SUBMISSIONS_URL.format(cik=cik)).json()
    recent = data["filings"]["recent"]
    for form, acc, doc, date in zip(
        recent["form"], recent["accessionNumber"], recent["primaryDocument"], recent["filingDate"]
    ):
        if form == "10-K":
            return acc.replace("-", ""), doc, date
    raise RuntimeError(f"No 10-K found for CIK {cik}")


def html_to_text(html: str) -> str:
    """Convert filing HTML to text, preserving table structure as pipe rows."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "head"]):
        tag.decompose()

    # Replace each table with a marked, pipe-delimited block so the sentence
    # splitter can treat it as one atomic unit instead of shredding it.
    for table in soup.find_all("table"):
        rows: list[str] = []
        for tr in table.find_all("tr"):
            cells = [
                re.sub(r"\s+", " ", td.get_text(" ", strip=True))
                for td in tr.find_all(["td", "th"])
            ]
            cells = [c for c in cells if c]
            if cells:
                rows.append(" | ".join(cells))
        table.replace_with(
            "\n\n" + TABLE_OPEN + "\n" + "\n".join(rows) + "\n" + TABLE_CLOSE + "\n\n"
            if rows
            else "\n\n"
        )

    # Block elements become paragraph breaks.
    for tag in soup.find_all(["p", "div", "br", "li", "h1", "h2", "h3", "h4", "tr"]):
        tag.append("\n")

    text = soup.get_text()
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch(
    tickers: tuple[str, ...] = DEFAULT_TICKERS, out_dir: Path | None = None
) -> list[Path]:
    out = out_dir or CORPUS_DIR
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    with _client() as client:
        ciks = _resolve_ciks(client, tickers)
        for ticker, cik in ciks.items():
            dest = out / f"{ticker}_10K.txt"
            if dest.exists() and dest.stat().st_size > 50_000:
                print(f"[sec] {ticker}: cached ({dest.stat().st_size // 1024} KB)")
                written.append(dest)
                continue

            acc, doc, date = _latest_10k(client, cik)
            url = ARCHIVE_URL.format(cik=cik, acc=acc, doc=doc)
            print(f"[sec] {ticker}: fetching 10-K filed {date}")
            html = client.get(url).text
            text = html_to_text(html)

            header = f"# {ticker} Form 10-K (filed {date})\nSource: {url}\n\n"
            dest.write_text(header + text, encoding="utf-8")
            print(f"[sec] {ticker}: wrote {len(text) // 1024} KB of text -> {dest.name}")
            written.append(dest)
            time.sleep(0.2)  # stay well under EDGAR's 10 req/s guidance

    manifest = out / "manifest.json"
    manifest.write_text(
        json.dumps({"tickers": list(ciks), "files": [p.name for p in written]}, indent=2),
        encoding="utf-8",
    )
    return written


if __name__ == "__main__":
    fetch()
