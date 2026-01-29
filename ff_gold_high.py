#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from urllib.request import Request, urlopen

FF_XML_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"

HIGH_IMPACT_VALUES = {"High"}

# Gold is driven mostly by USD macro; add more if you want
DRIVER_CURRENCIES = {"USD"}  # e.g. {"USD", "EUR", "GBP", "JPY"}
GOLD_TITLE_RE = re.compile(r"\b(gold|xau|bullion|precious metal)\b", re.I)


@dataclass
class EventRow:
    dt_utc: Optional[datetime]
    date_raw: str
    time_raw: str
    title: str
    currency: str
    impact: str
    actual: str
    forecast: str
    previous: str
    event_id: str


def _text(elem: Optional[ET.Element]) -> str:
    if elem is None or elem.text is None:
        return ""
    return elem.text.strip()


def _parse_dt_utc(date_str: str, time_str: str) -> Optional[datetime]:
    if not date_str or not time_str:
        return None
    if time_str.lower() in {"all day", "tentative"}:
        return None

    year = datetime.now(timezone.utc).year
    dt_str = f"{date_str} {year} {time_str}".strip()

    for fmt in ("%b %d %Y %I:%M%p", "%b %d %Y %I%p"):
        try:
            return datetime.strptime(dt_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    return None


def fetch_ff_xml(url: str = FF_XML_URL) -> bytes:
    req = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        },
    )
    with urlopen(req, timeout=30) as resp:
        return resp.read()


def parse_events(xml_bytes: bytes) -> list[EventRow]:
    root = ET.fromstring(xml_bytes)
    out: list[EventRow] = []

    for ev in root.findall(".//event"):
        title = _text(ev.find("title"))
        currency = _text(ev.find("currency")).upper()
        impact = _text(ev.find("impact"))
        actual = _text(ev.find("actual"))
        forecast = _text(ev.find("forecast"))
        previous = _text(ev.find("previous"))
        event_id = _text(ev.find("id")) or _text(ev.find("eventid"))

        date_raw = _text(ev.find("date"))
        time_raw = _text(ev.find("time"))
        dt_utc = _parse_dt_utc(date_raw, time_raw)

        out.append(
            EventRow(
                dt_utc=dt_utc,
                date_raw=date_raw,
                time_raw=time_raw,
                title=title,
                currency=currency,
                impact=impact,
                actual=actual,
                forecast=forecast,
                previous=previous,
                event_id=event_id,
            )
        )

    return out


def is_relevant_for_gold(row: EventRow) -> bool:
    # High-impact USD (and others you choose) is the core driver set
    if row.impact in HIGH_IMPACT_VALUES and row.currency in DRIVER_CURRENCIES:
        return True

    # Also keep rare explicit gold mentions
    if GOLD_TITLE_RE.search(row.title):
        return True

    return False


def main():
    xml_bytes = fetch_ff_xml()
    rows = parse_events(xml_bytes)
    filtered = [r for r in rows if is_relevant_for_gold(r)]

    # Sort by known datetime; unknown at bottom
    filtered.sort(key=lambda r: r.dt_utc or datetime.max.replace(tzinfo=timezone.utc))

    w = csv.writer(sys.stdout, delimiter="\t")
    w.writerow(
        [
            "datetime_utc",
            "date_raw",
            "time_raw",
            "currency",
            "impact",
            "title",
            "actual",
            "forecast",
            "previous",
            "event_id",
        ]
    )
    for r in filtered:
        w.writerow(
            [
                r.dt_utc.isoformat(timespec="seconds") if r.dt_utc else "",
                r.date_raw,
                r.time_raw,
                r.currency,
                r.impact,
                r.title,
                r.actual,
                r.forecast,
                r.previous,
                r.event_id,
            ]
        )


if __name__ == "__main__":
    main()