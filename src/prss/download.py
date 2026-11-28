"""Download a large file, resuming after a dropped connection instead of starting again.

First written for PanNuke (script 10), where the Warwick server dropped the connection every
minute or two; now shared with the TCGA slide download (script 21).

How resuming works: an HTTP "Range: bytes=N-" header asks the server for the file from byte N
onwards. What has arrived is kept in a .part file, and after a drop we ask only for the rest.
The .part file gets its real name only once `check(path)` passes (for example "is a valid
zip"), so a broken download never looks finished.
"""
import time

import requests


def fetch(url, dest, check=None, tries=30, label=""):
    """Download url to dest (a Path). Returns dest. Raises SystemExit after `tries` failures."""
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    label = label or dest.name
    for attempt in range(1, tries + 1):
        have = part.stat().st_size if part.exists() else 0
        try:
            with requests.get(url, stream=True, timeout=60,
                              headers={"Range": f"bytes={have}-"} if have else {}) as r:
                if r.status_code == 416:           # "range not satisfiable": we already have it all
                    break
                r.raise_for_status()
                if have and r.status_code != 206:  # server ignored the Range header: start over
                    have = 0
                total = have + int(r.headers.get("Content-Length", 0))
                with open(part, "ab" if have else "wb") as fh:
                    for i, chunk in enumerate(r.iter_content(chunk_size=1 << 20)):
                        fh.write(chunk)
                        if i % 100 == 0:
                            print(f"  {label}: {fh.tell() / 1e6:.0f} / {total / 1e6:.0f} MB", flush=True)
            break
        except (requests.ConnectionError, requests.exceptions.ChunkedEncodingError, requests.Timeout) as e:
            got = part.stat().st_size if part.exists() else 0
            print(f"  {label}: connection dropped at {got / 1e6:.0f} MB ({type(e).__name__}); "
                  f"retry {attempt}/{tries} in 5 s")
            time.sleep(5)
    else:
        raise SystemExit(f"{label}: gave up after {tries} tries; run the script again to resume")
    if check is not None and not check(part):
        raise SystemExit(f"{label}: download finished but the file is not valid. Delete {part} and run again.")
    part.rename(dest)
    return dest
