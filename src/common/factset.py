"""The canonical fact set — the ONLY determinant of the report text.

The text layer may vary word order and phrasing; it can never add or drop a fact.
`facts_hash` makes that auditable: an identical hash means identical clinical
content, independently of language and of which surface variant was rendered.
"""
from __future__ import annotations

import hashlib
import json


def build_factset(rec: dict) -> dict:
    """Build the canonical fact set from a parsed label record."""
    stones = rec.get("stones", [])
    kidneys: dict[str, dict] = {}
    for side, key in (("RIGHT", "right"), ("LEFT", "left")):
        ss = [s for s in stones if s["side"] == side]
        ss_sorted = sorted(
            ss, key=lambda s: (-(s["size_mm"] or -1), s["zone"])
        )
        mms = [s["size_mm"] for s in ss if s.get("size_known")]
        kidneys[key] = {
            "present": bool(ss) or rec["laterality"] in ("BILATERAL", side),
            "n_characterized": len(ss),
            "stones": [
                {
                    "zone": s["zone"],
                    "mm": s["size_mm"],
                    "cls": s["size_class"],
                }
                for s in ss_sorted
            ],
            "max_mm": max(mms) if mms else None,
            "max_cls": (
                max(ss_sorted, key=lambda s: s["size_mm"] or -1)["size_class"] if mms else None
            ),
            "zones": sorted({s["zone"] for s in ss}),
        }

    with_mm = [s for s in stones if s.get("size_known")]
    largest = max(with_mm, key=lambda s: s["size_mm"]) if with_mm else None

    qualifier = rec.get("count_qualifier", "EXACT")
    # Invariant: when the report says "numerous" there is NO exact count.
    # Enforcing that here structurally prevents a fact from surviving in the fact
    # set without appearing in the text, which would break the round-trip gate.
    total_n = None if qualifier == "MANY" else rec.get("n_stones_effective")

    return {
        "laterality": rec["laterality"],
        "total_n": total_n,
        "total_qualifier": qualifier,
        "n_characterized": len(stones),
        "kidneys": kidneys,
        "largest": (
            {
                "side": largest["side"],
                "zone": largest["zone"],
                "mm": largest["size_mm"],
                "cls": largest["size_class"],
            }
            if largest
            else None
        ),
        "anomalies": sorted(rec.get("anomalies", [])),
    }


def facts_hash(fs: dict) -> str:
    return hashlib.sha1(
        json.dumps(fs, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]


def canonical_stone_tuples(fs: dict) -> list[tuple]:
    """The unit of evaluation: (side, zone, size_class), used to compare a model's
    output against the reference."""
    out = []
    for side, key in (("RIGHT", "right"), ("LEFT", "left")):
        for s in fs["kidneys"][key]["stones"]:
            out.append((side, s["zone"], s["cls"]))
    return sorted(out)
