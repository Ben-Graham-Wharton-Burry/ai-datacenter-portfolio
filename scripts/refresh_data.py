"""Refresh the frozen data snapshot. Run on your own machine (needs internet):

    python scripts/refresh_data.py                 # everything
    python scripts/refresh_data.py --only caps,cpi # just some sources

Downloads, from 2019-01-01 to today:
  * daily total-return prices for every ticker in src/config.py (Yahoo Finance)
  * Fama-French five factors and the risk-free rate, monthly (Kenneth French's library)
  * US CPI, all items, seasonally adjusted, monthly (FRED series CPIAUCSL)

and overwrites the files in data_cache/. The notebooks and build_exhibits.py
read only these files, so results change only when you choose to refresh.

Each source is saved as soon as it arrives, so one failure does not lose the
others. If a download fails (a firewall, a changed URL, a bot check), the
script saves what the server sent to data_cache/_debug_<source>.txt and keeps
the previous file. As a fallback, download the file in your browser and put
it in data_cache/manual/ under the name printed by the script; the next run
uses it.
"""

from __future__ import annotations

import io
import json
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CACHE = ROOT / "data_cache"

FF5_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
           "F-F_Research_Data_5_Factors_2x3_CSV.zip")
CPI_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"


MANUAL = CACHE / "manual"


def _get(url: str) -> tuple[bytes, str, str]:
    """Body, final URL after redirects, and content type."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read(), r.geturl(), r.headers.get("Content-Type", "?")


def _diagnose(name: str, body: bytes, final_url: str, ctype: str) -> None:
    """Say what came back instead of the expected file, and keep a copy to inspect."""
    CACHE.mkdir(exist_ok=True)
    debug = CACHE / f"_debug_{name}.txt"
    debug.write_bytes(body)
    head = body[:300].decode("latin-1", "replace").replace("\n", " ")
    print(f"  Unexpected response: {len(body):,} bytes, content type {ctype}")
    print(f"  Final URL: {final_url}")
    print(f"  Starts with: {head[:200]}")
    print(f"  Saved to {debug.relative_to(ROOT)}")


def _ff5_text() -> str | None:
    """The five-factor CSV text: downloaded, or a manual copy (zip or csv) in data_cache/manual/."""
    try:
        body, final_url, ctype = _get(FF5_URL)
        if body[:2] == b"PK":                                   # every zip file starts with these bytes
            z = zipfile.ZipFile(io.BytesIO(body))
            return z.read(z.namelist()[0]).decode("latin-1")
        _diagnose("ff5", body, final_url, ctype)
    except Exception as e:                                      # network or certificate errors
        print(f"  Download failed: {type(e).__name__}: {e}")
    for f in (MANUAL / "F-F_Research_Data_5_Factors_2x3_CSV.zip", MANUAL / "F-F_Research_Data_5_Factors_2x3.csv"):
        if f.exists():
            print(f"  Using manual copy {f.relative_to(ROOT)}")
            if f.suffix == ".zip":
                z = zipfile.ZipFile(f)
                return z.read(z.namelist()[0]).decode("latin-1")
            return f.read_text(encoding="latin-1")
    print(f"  Fallback: download {FF5_URL}\n  in your browser and save it as "
          f"data_cache/manual/F-F_Research_Data_5_Factors_2x3_CSV.zip")
    return None


def _cpi_text() -> str | None:
    try:
        body, final_url, ctype = _get(CPI_URL)
        text = body.decode("utf-8", "replace")
        if text.startswith(("DATE", "observation_date")):       # FRED's CSV header
            return text
        _diagnose("cpi", body, final_url, ctype)
    except Exception as e:
        print(f"  Download failed: {type(e).__name__}: {e}")
    f = MANUAL / "CPIAUCSL.csv"
    if f.exists():
        print(f"  Using manual copy {f.relative_to(ROOT)}")
        return f.read_text()
    print(f"  Fallback: download {CPI_URL}\n  in your browser and save it as data_cache/manual/CPIAUCSL.csv")
    return None


def parse_ff5(text: str) -> pd.DataFrame:
    """The monthly block of Kenneth French's five-factor CSV, in decimals.

    The file has a few lines of description, a header row starting with a
    comma, monthly rows keyed YYYYMM, a blank line, then an annual block
    keyed YYYY, which is ignored.
    """
    rows, header, started = [], None, False
    for line in text.splitlines():
        cells = [c.strip() for c in line.split(",")]
        if not started:
            if len(cells) > 1 and cells[0] == "" and "Mkt-RF" in cells:
                header, started = cells[1:], True
            continue
        if not cells[0]:
            if rows:
                break                      # blank line ends the monthly block
            continue
        if len(cells[0]) != 6 or not cells[0].isdigit():
            break                          # the annual block (YYYY) begins
        rows.append([cells[0]] + [float(x) / 100 for x in cells[1:len(header) + 1]])
    df = pd.DataFrame(rows, columns=["ym"] + header)
    df.index = pd.to_datetime(df.pop("ym"), format="%Y%m") + pd.offsets.MonthEnd(0)
    return df


def parse_cpi(text: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    date_col, val_col = df.columns[0], df.columns[1]
    s = pd.Series(pd.to_numeric(df[val_col], errors="coerce").values,
                  index=pd.to_datetime(df[date_col]) + pd.offsets.MonthEnd(0), name="CPI")
    return s.dropna()


def _naive(idx) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    return idx.tz_convert(None) if idx.tz is not None else idx


def refresh_prices(C, yf) -> dict:
    tickers = sorted(set(C.ALL_TICKERS + [C.BENCHMARK, "IWD", "IWF", "IWM"]))
    print(f"Prices: {len(tickers)} tickers from Yahoo Finance...")
    px = yf.download(tickers, start=C.START_DATE, auto_adjust=True, progress=False)["Close"]
    px.index = _naive(px.index)
    px = px.dropna(how="all")
    missing = [t for t in tickers if t not in px or px[t].isna().all()]
    if missing:
        raise SystemExit(f"No prices for {missing}; snapshot left unchanged.")
    print(f"  {px.index[0]:%Y-%m-%d} to {px.index[-1]:%Y-%m-%d}, {len(px)} days")
    px.to_csv(CACHE / "prices_daily.csv")
    return {"prices": f"{px.index[-1]:%Y-%m-%d}", "end": f"{px.index[-1]:%Y-%m-%d}",
            "rows": int(len(px)), "columns": list(px.columns)}


def refresh_ff5() -> dict:
    print("Fama-French five factors...")
    text = _ff5_text()
    if not text:
        print("  Kept the previous ff5_monthly.csv.")
        return {}
    ff = parse_ff5(text)
    ff.to_csv(CACHE / "ff5_monthly.csv")
    print(f"  {ff.index[0]:%Y-%m} to {ff.index[-1]:%Y-%m}")
    return {"ff5": f"{ff.index[-1]:%Y-%m}"}


def refresh_cpi() -> dict:
    print("US CPI (FRED CPIAUCSL)...")
    text = _cpi_text()
    if not text:
        print("  Kept the previous cpi_monthly.csv (if any).")
        return {}
    cpi = parse_cpi(text)
    cpi.to_frame().to_csv(CACHE / "cpi_monthly.csv")
    print(f"  {cpi.index[0]:%Y-%m} to {cpi.index[-1]:%Y-%m}")
    return {"cpi": f"{cpi.index[-1]:%Y-%m}"}


def refresh_caps(C, yf) -> dict:
    """Shares outstanding x unadjusted close on 31 Dec 2024, for the Black-Litterman prior."""
    from src import estimators as ES, market_caps as MC
    names = list(C.SINGLE_NAMES)
    print(f"Market caps on {MC.AS_OF}: {len(names)} names...")
    raw = yf.download(names, start="2024-12-16", end="2025-01-03", auto_adjust=False, progress=False)["Close"]
    raw.index = _naive(raw.index)
    close = raw.loc[:MC.AS_OF].ffill().iloc[-1]                 # last close on or before 31 Dec 2024
    rows = []
    for t in names:
        ref = float(ES.REFERENCE_CAP_2024[t])
        row = {"ticker": t, "shares": float("nan"), "shares_date": "", "price": float(close.get(t, float("nan"))),
               "reference_bn": ref}
        try:
            sh = yf.Ticker(t).get_shares_full(start="2024-06-01", end="2025-06-30")
            if sh is None or len(sh) == 0:
                raise ValueError("no share-count history")
            sh = pd.Series(sh.values, index=_naive(sh.index)).sort_index()
            sh = sh[~sh.index.duplicated(keep="last")]
            before = sh.loc[:MC.AS_OF]
            when, shares = (before.index[-1], before.iloc[-1]) if len(before) else (sh.index[0], sh.iloc[0])
            row.update(shares=float(shares), shares_date=f"{when:%Y-%m-%d}")
            row.update(MC.resolve(t, float(shares), row["price"], ref))
        except Exception as e:                                  # keep going; the name falls back to its reference
            row.update(market_cap_bn=float("nan"), unit="missing", deviation=float("nan"),
                       flag=f"review: {type(e).__name__}: {e}")
        rows.append(row)
    df = pd.DataFrame(rows).set_index("ticker")
    df.to_csv(MC.PATH)
    show = df[["market_cap_bn", "reference_bn", "deviation", "unit", "flag"]].copy()
    show["market_cap_bn"] = show["market_cap_bn"].round(1)
    show["deviation"] = show["deviation"].map(lambda v: "" if pd.isna(v) else f"{v:+.0%}")
    with pd.option_context("display.width", 160, "display.max_colwidth", 60):
        print(show.to_string())
    n_flag = int((df["flag"].fillna("") != "").sum())
    print(f"  Saved data_cache/{MC.PATH.name}; {n_flag} name(s) flagged for review.")
    return {"caps": MC.AS_OF}


SOURCES = ("prices", "ff5", "cpi", "caps")


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Refresh the data snapshot in data_cache/.")
    ap.add_argument("--only", default=",".join(SOURCES),
                    help="comma-separated subset of: " + ", ".join(SOURCES) + " (default: all)")
    want = [x.strip() for x in ap.parse_args(argv).only.split(",") if x.strip()]
    bad = [x for x in want if x not in SOURCES]
    if bad:
        raise SystemExit(f"Unknown source(s) {bad}; choose from {SOURCES}.")

    from src import config as C
    CACHE.mkdir(exist_ok=True)
    MANUAL.mkdir(exist_ok=True)
    yf = None
    if "prices" in want or "caps" in want:
        import yfinance as yf

    meta_path = CACHE / "snapshot_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    refreshed = {}
    if "prices" in want:
        info = refresh_prices(C, yf)
        meta.update(end=info.pop("end"), rows=info.pop("rows"), columns=info.pop("columns"))
        refreshed.update(info)
    if "ff5" in want:
        refreshed.update(refresh_ff5())
    if "cpi" in want:
        refreshed.update(refresh_cpi())
    if "caps" in want:
        refreshed.update(refresh_caps(C, yf))

    meta.update({"start": C.START_DATE,
                 "source": "Yahoo Finance via yfinance; Kenneth French data library; FRED CPIAUCSL",
                 "adjustment": "auto_adjust=True (total return); market caps use unadjusted closes",
                 "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    meta["refreshed"] = {**meta.get("refreshed", {}), **refreshed}
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Snapshot updated in data_cache/: {refreshed}")
    missed = [x for x in want if x not in refreshed]
    if missed:
        print(f"Not refreshed: {missed}; see the messages above.")


if __name__ == "__main__":
    main()
