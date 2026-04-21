from __future__ import print_function

import os
import threading
from datetime import datetime
from inspect import _void
from typing import Dict

import requests

URL = "https://www.lottoaruba.com/templates/lotto/xjpot-tab.php"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:149.0) Gecko/20100101 Firefox/149.0",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://www.lottoaruba.com/?results=1&x=1",
}

data_htmls: Dict[int, str] = {}
data_rows: Dict[int, list[str]] = {}

stop_event = threading.Event()
page_lock = threading.Lock()
htmls_lock = threading.Lock()
rows_lock = threading.Lock()

next_page = 0


def fetch_page(page: int, game: int = 1) -> bool:
    response = requests.post(
        URL,
        headers=HEADERS,
        data={
            "i": game,
            "p": page,
        },
        timeout=30,
    )
    response.raise_for_status()

    if "No results found." in response.text:
        print(f"No results found for page {page} for game {game}")
        return False

    print(f"Fetched page {page} for game {game}")

    with htmls_lock:
        data_htmls[page] = response.text

    return True


def get_next_page() -> int:
    global next_page

    with page_lock:
        page = next_page
        next_page += 1
        return page


def worker(game: int = 0) -> None:
    while not stop_event.is_set():
        page = get_next_page()

        try:
            ok = fetch_page(page, game)
        except Exception as exc:
            print(f"Error on page {page}: {exc}")
            stop_event.set()
            return

        if not ok:
            stop_event.set()
            return


def process_html(html: str):
    rows = []
    parts = html.split("<tr")

    # Skip the first part which is before the first <tr>
    for part in parts[1:]:
        row = "<tr" + part
        if 'class="tr_' in row:
            rows.append(row.split("</tr>", 1)[0] + "</tr>")

    # Convert to CSV
    csv_rows = []
    for row in rows:
        cells = []
        for cell in row.split("<td")[1:]:
            cell_content = cell.split(">", 1)[1].split("</td>", 1)[0].strip()
            cells.append(cell_content)
        csv_rows.append(",".join(cells))

    # Append to rows dictionary
    with rows_lock:
        data_rows[len(data_rows)] = csv_rows

    return


if __name__ == "__main__":
    os.makedirs("tmp", exist_ok=True)

    thread_count = 10
    threads = []

    # Start worker threads to fetch pages concurrently
    for _ in range(thread_count):
        thread = threading.Thread(target=worker, args=(1,))
        thread.start()
        threads.append(thread)

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    # Process and print the number of rows for each fetched page
    for page in sorted(data_htmls):
        process_html(data_htmls[page])

    # Set today to YYYY_MM_DD
    today = datetime.now().strftime("%Y_%m_%d")

    # Store at tmp/{today}_xjpot.csv
    output_file = f"tmp/{today}_xjpot.csv"
    with open(output_file, "w") as f:
        for page in sorted(data_rows):
            for row in data_rows[page]:
                f.write(row + "\n")
