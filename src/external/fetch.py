"""Download the external test data into data/external/ (gitignored, never committed).

    python3 -m src.external.fetch [abt_buy] [kitspace] [jlcpcb] [ca_po]     # no arguments = all four

Sources and their terms (checked 2026-11-14; see the README for the notes):
  abt_buy   University of Mannheim "Benchmark Matching Tasks" page; original data from the Leipzig database group.
  kitspace  1-click-bom.tsv files from open-hardware projects on GitHub; each repository has its own license.
  jlcpcb    JLCPCB parts catalog as mirrored by the yaqwsx/jlcparts project (about 400 MB download).
  ca_po     State of California "Purchase Order Data 2012-2015" from data.ca.gov, first rows via the public API.

Standard library only. Each source goes in its own directory. Files are treated as data, never executed.
"""
from __future__ import annotations

import json
import shutil
import struct
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib
from pathlib import Path

ROOT = Path("data/external")
UA = {"User-Agent": "bom-tie-out-external-tests"}

MANNHEIM = "http://data.dws.informatik.uni-mannheim.de/benchmarkmatchingtasks/data/abt-buy/"
KITSPACE = [  # (repository, path to 1-click-bom.tsv)
    ("ashtuchkin/vive-diy-position-sensor", "1-click-bom.tsv"), ("kanflo/aaduino", "hardware/1-click-bom.tsv"),
    ("emard/ulx3s", "1-click-bom.tsv"), ("im-tomu/tomu-hardware", "1-click-bom.tsv"), ("uCAN-LIN/LinUSBConverter", "pcb/1-click-bom.tsv"),
    ("julbouln/ice40_eink_controller", "hardware/1-click-bom.tsv"), ("jasiek/antenna-analyser", "1-click-bom.tsv"),
    ("BadenLab/Openspritzer", "1-click-bom.tsv"), ("mcqn/my-babys-got-led", "electronics/1-click-bom.tsv"),
    ("rwb27/openflexure_nano_motor_controller", "pcb_design/1-click-bom.tsv"), ("jpralves/cseduino", "boards/2-layer/1-click-bom.tsv"),
    ("weirdgyn/Driverino-Shield", "PCB/1-click-bom.tsv"), ("julbouln/ifusb", "hardware/1-click-bom.tsv"),
    ("lukecyca/partycat", "1-click-bom.tsv"), ("JarrettR/USBvil", "1-click-bom.tsv"), ("balthazar-space/Unifying-PCB", "1-click-bom.tsv"),
    ("UDXS/Arduitsy", "1-click-bom.tsv"), ("GaudiLabs/ChaosLooper", "1-click-bom.tsv"), ("derekenos/the_deMIDulator", "1-click-bom.tsv"),
    ("jasiek/ESP8266-PowerMonitor", "hardware/1-click-bom.tsv"), ("python-ugame/ugame-10-hardware", "bom/1-click-bom.tsv"),
    ("less-smog/hardware", "1-click-bom.tsv"), ("kitspace/jelly", "1-click-bom.tsv"), ("intergalaktik/ulx4m-ls", "1-click-bom.tsv"),
]
JLC = "https://yaqwsx.github.io/jlcparts/data/"
CA_API = "https://data.ca.gov/api/3/action/datastore_search"
CA_RESOURCE = "bb82edc5-9c78-44e2-8947-68ece26197c5"
CA_ROWS = 3000  # the test samples 500 of these


def get(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r, tmp.open("wb") as f:
            shutil.copyfileobj(r, f)
    except urllib.error.HTTPError as e:
        tmp.unlink(missing_ok=True)
        if e.code == 404:
            return False
        raise
    tmp.rename(dest)
    return True


def abt_buy():
    d = ROOT / "abt_buy"
    for name in ("gs_train.csv", "gs_val.csv", "gs_test.csv", "records.zip"):
        get(MANNHEIM + name, d / name)
    with zipfile.ZipFile(d / "records.zip") as z:
        for info in z.infolist():  # write members by base name only; never trust archive paths
            if not info.is_dir():
                (d / Path(info.filename).name).write_bytes(z.read(info))
    print("abt_buy:", sorted(p.name for p in d.iterdir()))


def kitspace():
    d, ok = ROOT / "kitspace", 0
    for repo, path in KITSPACE:
        ok += get(f"https://raw.githubusercontent.com/{repo}/HEAD/{urllib.parse.quote(path)}", d / (repo.replace("/", "__") + ".tsv"))
    print(f"kitspace: {ok} of {len(KITSPACE)} BOM files")


def jlcpcb():
    d = ROOT / "jlcpcb"
    if (d / "cache.sqlite3").exists():
        print("jlcpcb: cache.sqlite3 already present")
        return
    volumes = []
    for i in range(1, 100):  # split archive: cache.z01, cache.z02, ... then cache.zip
        if not get(f"{JLC}cache.z{i:02d}", d / f"cache.z{i:02d}"):
            break
        volumes.append(d / f"cache.z{i:02d}")
    get(JLC + "cache.zip", d / "cache.zip")
    # A split zip cannot be opened by zipfile. The archive holds one deflated member that starts at the
    # first volume, so read its local header and inflate the stream across the volumes in order.
    def stream():
        for v in [*volumes, d / "cache.zip"]:
            with v.open("rb") as f:
                while chunk := f.read(1 << 22):
                    yield chunk

    chunks = stream()
    buf = next(chunks)
    if buf[:4] == b"PK\x07\x08":  # spanning marker
        buf = buf[4:]
    assert buf[:4] == b"PK\x03\x04", "unexpected archive layout"
    method, name_len, extra_len = struct.unpack_from("<H", buf, 8)[0], *struct.unpack_from("<HH", buf, 26)
    assert method == 8, "expected a deflated member"
    inflate, buf = zlib.decompressobj(-15), buf[30 + name_len + extra_len:]
    with (d / "cache.sqlite3").open("wb") as out:
        while True:
            out.write(inflate.decompress(buf))
            if inflate.eof:
                break
            buf = next(chunks)
    for p in [*volumes, d / "cache.zip"]:
        p.unlink(missing_ok=True)
    print(f"jlcpcb: cache.sqlite3 {(d / 'cache.sqlite3').stat().st_size / 1e9:.1f} GB")


def ca_po():
    d = ROOT / "ca_po"
    get(f"{CA_API}?resource_id={CA_RESOURCE}&limit={CA_ROWS}", d / "purchase_orders.json")
    n = len(json.loads((d / "purchase_orders.json").read_text())["result"]["records"])
    print(f"ca_po: {n} purchase-order rows")


if __name__ == "__main__":
    todo = sys.argv[1:] or ["abt_buy", "kitspace", "ca_po", "jlcpcb"]
    for name in todo:
        {"abt_buy": abt_buy, "kitspace": kitspace, "jlcpcb": jlcpcb, "ca_po": ca_po}[name]()
