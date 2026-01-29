import json
import time
from datetime import date, timedelta

import requests
from bs4 import BeautifulSoup

BASE = "https://www.forexfactory.com/calendar"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-US,en;q=0.9",
}

def week_param(d: date) -> str:
    # FF uses e.g. "may25.2025" meaning week that starts on that Sunday
    return d.strftime("%b").lower() + str(d.day) + "." + str(d.year)

def sunday_of_week(d: date) -> date:
    # make Sunday the start
    return d - timedelta(days=(d.weekday() + 1) % 7)

def fetch_week(week: str) -> str:
    r = requests.get(BASE, params={"week": week}, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.text

def parse_events(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")

    # You must inspect FF HTML to finalize these selectors.
    # The calendar is a table; events are rows with time/currency/impact/detail.
    rows = soup.select("table.calendar__table tr")
    events = []

    for tr in rows:
        tds = tr.find_all("td")
        if not tds:
            continue

        text = " ".join(tr.get_text(" ", strip=True).split())
        if not text:
            continue

        # Placeholder: store raw row text until you refine parsing
        events.append({"raw": text})

    return events

def iter_weeks_2025():
    d = sunday_of_week(date(2025, 1, 1))
    end = date(2026, 1, 1)
    while d < end:
        yield d
        d += timedelta(days=7)

all_events = []
for d in iter_weeks_2025():
    wk = week_param(d)
    html = fetch_week(wk)
    events = parse_events(html)
    for e in events:
        e["week"] = wk
    all_events.extend(events)

    time.sleep(2.0)  # be polite

with open("ff_calendar_2025.json", "w", encoding="utf-8") as f:
    json.dump(all_events, f, ensure_ascii=False, indent=2)

print("saved", len(all_events), "rows")