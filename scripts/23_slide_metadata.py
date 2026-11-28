"""Step 23 (Track 2b): how was each slide scanned? Read it from the file header, without downloading.

Why: in the pilot, tiles from one hospital (A8) found other A8 tiles 99.8% of the time. Before
calling that "lab style", rule out a plain technical cause: a different magnification, pixel
size, scanner or compression changes how every tile looks.

How, without downloading 1 GB per slide: a TCGA slide (.svs) is a TIFF file. Its first
"page" carries a short text description written by the scanner software, e.g.
    Aperio Image Library v10.0.51 ... |AppMag = 40|MPP = 0.2520|ScanScope ID = SS1283|
    Date = 06/21/2011|Q = 70|...
We open the remote file through a small "file-like" object that turns every read into an HTTP
Range request for just those bytes. tifffile then reads only what it needs: usually a few
hundred kilobytes per slide instead of a gigabyte.

Output: printed table per site, data/interim/tcga_slide_metadata.csv, and
results/tcga_slide_metadata.json (small, so it can be committed).

Reads the slide list of script 20 (pilot or full). Needs: tifffile (in the slides extra).
"""
import io
import json
import re
import time

import pandas as pd
import requests
import tifffile

from prss.slides.data import ROOT


class HttpRangeFile(io.RawIOBase):
    """A read-only, seekable file over HTTP: every read is a Range request, cached in blocks."""

    BLOCK = 1 << 16                                     # fetch 64 KB at a time

    def __init__(self, url, size, session):
        self.url, self.size, self.session, self.pos, self.cache = url, size, session, 0, {}
        self.bytes_fetched = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence] + offset
        return self.pos

    def _block(self, i):
        if i not in self.cache:
            start = i * self.BLOCK
            end = min(start + self.BLOCK, self.size) - 1
            for attempt in range(5):
                try:
                    r = self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=60)
                    r.raise_for_status()
                    break
                except requests.RequestException:
                    if attempt == 4:
                        raise
                    time.sleep(3)
            if r.status_code != 206:
                raise IOError("server ignored the Range header; reading the header would download the whole slide")
            self.cache[i] = r.content
            self.bytes_fetched += len(r.content)
        return self.cache[i]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while len(out) < n:
            b = self._block((self.pos + len(out)) // self.BLOCK)
            off = (self.pos + len(out)) % self.BLOCK
            out += b[off:off + n - len(out)]
        self.pos += n
        return bytes(out)

    def readinto(self, buf):
        data = self.read(len(buf))
        buf[:len(data)] = data
        return len(data)


def parse_description(desc):
    """'...|AppMag = 40|MPP = 0.2520|...' -> {'AppMag': '40', 'MPP': '0.2520', ...}"""
    return {k.strip(): v.strip() for k, v in re.findall(r"\|?\s*([A-Za-z][\w .]*?)\s*=\s*([^|]*)", desc or "")}


def slide_header(url, size, session):
    fh = HttpRangeFile(url, size, session)
    with tifffile.TiffFile(io.BufferedReader(fh, buffer_size=1 << 16)) as tif:
        page = tif.pages[0]
        d = parse_description(page.description)
        info = dict(
            width=page.imagewidth, height=page.imagelength,
            compression=getattr(page.compression, "name", str(page.compression)),   # e.g. JPEG, APERIO_JP2000_RGB
            app_mag=d.get("AppMag"), mpp=d.get("MPP"), jpeg_quality=d.get("Q"),
            scanner_id=d.get("ScanScope ID"), scan_date=d.get("Date"),
            software=(page.description or "").split("\n")[0].split("|")[0][:60],
        )
    info["kb_read"] = round(fh.bytes_fetched / 1024)
    return info


if __name__ == "__main__":
    slides = pd.read_csv(ROOT / "data" / "interim" / "tcga_slides.csv", dtype=str)
    session = requests.Session()
    rows = []
    for n, s in enumerate(slides.itertuples(), 1):
        url = f"https://api.gdc.cancer.gov/data/{s.file_id}"
        try:
            info = slide_header(url, int(s.size), session)
        except Exception as e:                      # one unreadable header should not stop the rest
            info = dict(error=f"{type(e).__name__}: {e}"[:120])
        rows.append(dict(patient=s.patient, site=s.site, **info))
        print(f"[{n}/{len(slides)}] {s.patient}: " + ", ".join(f"{k}={v}" for k, v in info.items()))

    meta = pd.DataFrame(rows)
    meta.to_csv(ROOT / "data" / "interim" / "tcga_slide_metadata.csv", index=False)
    cols = [c for c in ["app_mag", "mpp", "jpeg_quality", "compression", "scanner_id"] if c in meta]
    print("\nper site: the values seen (count of slides)")
    for site, g in meta.groupby("site"):
        print(f"  {site}: " + "; ".join(f"{c} " + ", ".join(f"{v} ({k})" for v, k in g[c].astype(str).value_counts().items())
                                        for c in cols))
    out = ROOT / "results" / "tcga_slide_metadata.json"
    json.dump(meta.to_dict(orient="records"), open(out, "w"), indent=1)
    print(f"\nsaved {out.relative_to(ROOT)}")
