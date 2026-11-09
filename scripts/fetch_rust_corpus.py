#!/usr/bin/env python
# Scale the Rust corpus for stage-1 codec training: top crates.io crates by downloads,
# permissive licenses, extract all .rs. Target ~100 MB of real Rust. Resumable + logged.
import json, os, subprocess, tarfile, time, urllib.request

D = "/home/cobrajet/ncm-lab/data/rust/corpus"
os.makedirs(D, exist_ok=True)
HDRS = {"User-Agent": "ncm-lab research (cobrajet302gt) codec-corpus/1.0"}
PERMISSIVE = {"MIT", "Apache-2.0", "MIT OR Apache-2.0", "Apache-2.0 OR MIT", "BSD-3-Clause", "Zlib", "MIT/Apache-2.0", "0BSD", "ISC", "Unlicense OR MIT", "MPL-2.0", "MIT OR Zlib"}

def get(url):
    req = urllib.request.Request(url, headers=HDRS)
    return json.loads(urllib.request.urlopen(req, timeout=60).read())

done = set(d for d in os.listdir(D))
print(f"have {len(done)} crates", flush=True)
picked = 0
for page in range(1, 12):
    try:
        data = get(f"https://crates.io/api/v1/crates?sort=downloads&per_page=100&page={page}")
    except Exception as e:
        print("api fail", repr(e)[:80], flush=True); break
    for c in data.get("crates", []):
        name = c.get("name") or c["crate"]["name"]
        ver = c.get("max_stable_version") or c.get("default_version")
        if not ver or f"{name}-{ver}" in done or picked >= 220:
            continue
        try:
            meta = get(f"https://crates.io/api/v1/crates/{name}/{ver}")
            lic = meta["version"].get("license") or ""
        except Exception:
            continue
        if not any(l in lic for l in ("MIT", "Apache", "BSD", "Zlib", "0BSD", "ISC", "Unlicense", "MPL")):
            continue
        url = f"https://static.crates.io/crates/{name}/{name}-{ver}.crate"
        try:
            raw = urllib.request.urlopen(urllib.request.Request(url, headers=HDRS), timeout=120).read()
            if len(raw) > 40_000_000:
                continue
            open(f"/tmp/{name}.crate", "wb").write(raw)
            out = os.path.join(D, f"{name}-{ver}")
            with tarfile.open(f"/tmp/{name}.crate") as tf:
                tf.extractall(out)
            os.remove(f"/tmp/{name}.crate")
            done.add(f"{name}-{ver}")
            picked += 1
            if picked % 10 == 0:
                tot = subprocess.run(f"find {D} -name '*.rs' -exec cat {{}} + | wc -c", shell=True, capture_output=True).stdout
                print(f"picked {picked} crates, {int(tot)/1e6:.0f} MB rust, {time.strftime('%H:%M:%S')}", flush=True)
        except Exception as e:
            print("dl fail", name, repr(e)[:60], flush=True)
    time.sleep(1)
tot = subprocess.run(f"find {D} -name '*.rs' | wc -l; find {D} -name '*.rs' -exec cat {{}} + | wc -c", shell=True, capture_output=True).stdout.decode().split()
print(f"DONE crates={len(done)} files={tot[0]} bytes={tot[1]}", flush=True)
