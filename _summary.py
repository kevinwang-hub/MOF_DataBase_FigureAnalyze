import json
from collections import Counter

data = json.loads(open("parsed_output/test1\uff1a02439suppappendix/02439suppappendix/hybrid_auto/mof_analysis_results.json").read())
print(f"Total records: {len(data)}")

errors = [r for r in data if "error" in r]
success = [r for r in data if "error" not in r]
print(f"Success: {len(success)}")
print(f"Failed: {len(errors)}")
print()

types = Counter()
for r in success:
    fm = r.get("figure_metadata") or {}
    ft = fm.get("figure_type", "unknown")
    types[ft] += 1
print("Figure types:")
for k, v in types.most_common():
    print(f"  {k}: {v}")
print()

mats = set()
for r in success:
    m = (r.get("material") or {}).get("name")
    if m:
        mats.add(m)
print(f"Materials identified ({len(mats)}):")
for m in sorted(mats):
    print(f"  {m}")
print()

confs = Counter()
for r in success:
    fm = r.get("figure_metadata") or {}
    c = fm.get("confidence", "?")
    confs[c] += 1
print("Confidence:")
for k, v in confs.most_common():
    print(f"  {k}: {v}")
print()

arrays = [r for r in data if "raw_array" in r]
print(f"Array-type responses (tables/structures): {len(arrays)}")
print()

# BET values
bets = []
for r in success:
    ch = r.get("characterization") or {}
    b = ch.get("BET_surface_area_m2g")
    if b:
        name = (r.get("material") or {}).get("name", "?")
        img = (r.get("_meta") or {}).get("source_image", "?")
        bets.append((name, b, img))
if bets:
    print("BET surface areas found:")
    for name, b, img in bets:
        print(f"  {name}: {b} m2/g  ({img})")
print()

# Techniques
techs = Counter()
for r in success:
    ch = r.get("characterization") or {}
    for t in (ch.get("techniques_observed") or []):
        techs[t] += 1
if techs:
    print("Techniques observed:")
    for k, v in techs.most_common(15):
        print(f"  {k}: {v}")
