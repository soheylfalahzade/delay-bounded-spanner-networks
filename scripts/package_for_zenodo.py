#!/usr/bin/env python3
"""Bundle the data behind the reported numbers for a Zenodo deposit (DOI):
graph cache (OSM-derived, ODbL), raw benchmark CSVs, figures, SHA256SUMS and a .zenodo.json metadata draft.
    python scripts/package_for_zenodo.py
Output: data_archive/ (git-ignored). Upload the .tar.gz and .zenodo.json at https://zenodo.org/uploads/new,
or enable the GitHub-Zenodo integration and publish a GitHub release."""
import datetime, glob, hashlib, json, os, re, tarfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
out = os.path.join(ROOT, "data_archive")
os.makedirs(out, exist_ok=True)
patterns = ["results/graph_cache/*", "results/benchmarks/*.csv", "results/benchmarks/*.json", "results/benchmarks/*.md",
            "results/*.json", "results/*.csv", "results/*.md", "results/figures/*", "requirements*.txt", "CITATION.cff", "LICENSE"]
files = sorted({f for p in patterns for f in glob.glob(p) if os.path.isfile(f)})
stamp = datetime.date.today().isoformat()
tar_path = os.path.join(out, f"delay_bounded_spanner_data_{stamp}.tar.gz")
with tarfile.open(tar_path, "w:gz") as tf:
    for f in files:
        tf.add(f)
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()
with open(os.path.join(out, "SHA256SUMS.txt"), "w") as fh:
    fh.write(f"{sha(tar_path)}  {os.path.basename(tar_path)}\n")
    for f in files:
        fh.write(f"{sha(f)}  {f}\n")
authors = []
if os.path.exists("CITATION.cff"):
    cur = None
    for line in open("CITATION.cff", encoding="utf-8"):
        m = re.match(r"\s*-?\s*(family-names|given-names):\s*\"?([^\n\"]+)\"?\s*$", line)
        if m:
            if m.group(1) == "family-names":
                cur = {"fam": m.group(2).strip(), "giv": ""}
                authors.append(cur)
            elif cur is not None:
                cur["giv"] = m.group(2).strip()
        elif re.match(r"^\S", line) and not line.startswith("authors"):
            cur = None
    authors = [{"name": (a["fam"] + (", " + a["giv"] if a["giv"] else ""))} for a in authors]
meta = {"title": "Data and graph snapshots for: Delay-bounded time-varying geometric t-spanners for urban road networks",
        "upload_type": "dataset", "description": "OpenStreetMap-derived city street-network snapshots (ODbL), benchmark outputs, figures and checksums behind the reported results.",
        "creators": authors or [{"name": "FILL IN, Name"}], "license": "odc-odbl",
        "keywords": ["geometric spanners", "road networks", "time-dependent travel times", "OpenStreetMap"],
        "related_identifiers": [{"relation": "isSupplementTo", "identifier": "https://github.com/soheylfalahzade/delay-bounded-spanner-networks", "scheme": "url"}]}
json.dump(meta, open(os.path.join(out, ".zenodo.json"), "w"), indent=2)
print(f"{len(files)} files -> {tar_path} ({os.path.getsize(tar_path) / 1e6:.1f} MB); creators: {meta['creators']}")
