"""Recover the fact set from report text — the exact inverse of templates.py.

It serves two purposes:

  1) The hard gate of stage 10. Every generated report must parse back into the
     identical fact set. If it does not, the template is ambiguous, and that is
     fixed before any model is trained rather than discovered afterwards.
  2) Model evaluation. A report generated from the model's own predictions is
     parsed and compared against the reference fact set, which is what the
     field-level F1 and the hallucination and omission rates are computed on.

ONLY the findings section is parsed. The impression section restates the largest
stone, so parsing it would double-count that stone in the list.
"""
from __future__ import annotations

import re
from re import error

ZONE_TR_MAP = {"üst": "UPPER", "orta": "MID", "alt": "LOWER",
               "orta-üst": "MID_UPPER", "orta-alt": "MID_LOWER"}
ZONE_EN_MAP = {"upper zone": "UPPER", "mid zone": "MID", "lower zone": "LOWER",
               "mid-to-upper zone": "MID_UPPER", "mid-to-lower zone": "MID_LOWER"}
CLS_TR_MAP = {"mikrolitiyazis": "MIKROLITIYAZIS", "küçük taş": "KUCUK",
              "orta boy taş": "ORTA", "büyük taş": "BUYUK", "çok büyük taş": "COK_BUYUK"}
CLS_EN_MAP = {"microlithiasis": "MIKROLITIYAZIS", "small": "KUCUK", "medium-sized": "ORTA",
              "large": "BUYUK", "very large": "COK_BUYUK"}

_NUM = r"\d+(?:[.,]\d+)?"
_CLAUSE_TR = re.compile(
    rf"(?P<zone>orta-üst|orta-alt|üst|orta|alt)\s*zonda\s+"
    rf"(?:(?P<mm1>{_NUM})\s*mm\s*boyutunda|boyutu\s*(?P<mm2>{_NUM})\s*mm\s*ölçülen"
    rf"|(?P<mm3>{_NUM})\s*mm\s*çapında|boyutu\s*ölçülemeyen)\s*"
    rf"\((?P<cls>mikrolitiyazis|küçük taş|orta boy taş|büyük taş|çok büyük taş|boyutu bilinmiyor)\)",
    re.IGNORECASE,
)
_ZONES_EN = r"mid-to-upper zone|mid-to-lower zone|upper zone|mid zone|lower zone"
_CLS_EN = r"microlithiasis|medium-sized|very large|small|large|of unknown size"
_CLAUSE_EN = re.compile(
    rf"(?:"
    rf"an?\s+(?P<mm1>{_NUM})\s*mm\s+calculus\s*\((?P<cls1>{_CLS_EN})\)\s*in\s+the\s+(?P<zone1>{_ZONES_EN})"
    rf"|an?\s+calculus\s+in\s+the\s+(?P<zone2>{_ZONES_EN})(?:\s+of\s+the\s+\w+\s+kidney)?"
    rf"\s+measuring\s+(?P<mm2>{_NUM})\s*mm\s*\((?P<cls2>{_CLS_EN})\)"
    rf"|an?\s+calculus\s+of\s+indeterminate\s+size\s*\((?P<cls3>{_CLS_EN})\)\s*in\s+the\s+(?P<zone3>{_ZONES_EN})"
    rf")",
    re.IGNORECASE,
)


def _pick(m: re.Match, names: tuple[str, ...]) -> str | None:
    """Return the group of whichever alternative form matched."""
    for g in names:
        try:
            v = m.group(g)
        except (IndexError, error):
            continue
        if v:
            return v
    return None


def _mm(m: re.Match) -> float | None:
    v = _pick(m, ("mm1", "mm2", "mm3"))
    return float(v.replace(",", ".")) if v else None


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=\.)\s+", text) if s.strip()]


def _findings_section(text: str) -> str:
    """Extract only the findings section from a complete report."""
    m = re.search(r"(?:BULGULAR|FINDINGS)\s*:\s*(.*?)(?:\n\s*\n|\Z)", text, re.S | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.split(r"\n\s*(?:SONUÇ|IMPRESSION)\s*:", text, flags=re.IGNORECASE)[0]
    return m.strip()


def _impression_section(text: str) -> str:
    m = re.search(r"(?:SONUÇ|IMPRESSION)\s*:\s*(.*)\Z", text, re.S | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def parse_report(text: str, lang: str) -> dict:
    """Metinden FactSet'i geri kur. `lang` 'tr' veya 'en'."""
    findings = _findings_section(text)
    impression = _impression_section(text)
    tr = lang == "tr"
    clause_re = _CLAUSE_TR if tr else _CLAUSE_EN
    zone_map = ZONE_TR_MAP if tr else ZONE_EN_MAP
    cls_map = CLS_TR_MAP if tr else CLS_EN_MAP
    side_pat = (r"(sağ|sol)\s*böbre", r"(right|left)\s*kidney")[0 if tr else 1]

    kid: dict[str, dict] = {
        "right": {"present": False, "stones": []},
        "left": {"present": False, "stones": []},
    }
    anomalies: set[str] = set()
    total_n = None
    qualifier = "EXACT"

    for sent in _split_sentences(findings):
        low = sent.lower()

        if tr:
            if "at nalı böbrek" in low:
                anomalies.add("HORSESHOE_KIDNEY")
            if "tek bir büyük taş izlenimi" in low:
                anomalies.add("MANY_SMALL_STONES_MIMIC_LARGE")
            if "piramidlerde hiperdens" in low:
                anomalies.add("HYPERDENSE_PYRAMID_MIMIC")
            if "komplike kist" in low:
                anomalies.add("COMPLICATED_CYST_HYPERDENSE_LESION")
            if "üreterde taş" in low:
                anomalies.add("URETERAL_STONE_PRESENT")
            if "mesane divertikülü" in low:
                anomalies.add("BLADDER_DIVERTICULUM_STONE")
            if "sayılamayacak kadar çok sayıda" in low:
                qualifier = "MANY"
            m = re.search(r"toplam\s+(\d+)\s*adet", low)
            if m:
                total_n = int(m.group(1))
        else:
            if "horseshoe kidney" in low:
                anomalies.add("HORSESHOE_KIDNEY")
            if "simulate a single large calculus" in low:
                anomalies.add("MANY_SMALL_STONES_MIMIC_LARGE")
            if "hyperdense renal pyramids" in low:
                anomalies.add("HYPERDENSE_PYRAMID_MIMIC")
            if "complicated cyst" in low:
                anomalies.add("COMPLICATED_CYST_HYPERDENSE_LESION")
            if "ureteral calculus" in low:
                anomalies.add("URETERAL_STONE_PRESENT")
            if "bladder diverticulum" in low:
                anomalies.add("BLADDER_DIVERTICULUM_STONE")
            if "too many to count" in low:
                qualifier = "MANY"
            m = re.search(r"total of\s+(\d+)\s+calculi", low)
            if m:
                total_n = int(m.group(1))

        ms = re.search(side_pat, low)
        if not ms:
            continue
        side_key = "right" if ms.group(1) in ("sağ", "right") else "left"
        kid[side_key]["present"] = True
        for m in clause_re.finditer(sent):
            zone_raw = (m.group("zone") if tr else _pick(m, ("zone1", "zone2", "zone3"))).lower()
            cls_raw = (m.group("cls") if tr else _pick(m, ("cls1", "cls2", "cls3"))).lower()
            kid[side_key]["stones"].append({
                "zone": zone_map[zone_raw],
                "mm": _mm(m),
                "cls": cls_map.get(cls_raw),
            })

    if total_n is None:
        low = impression.lower()
        m = re.search(r"toplam\s+(\d+)\s*adet" if tr else r"total of\s+(\d+)\s+renal calculi", low)
        if m:
            total_n = int(m.group(1))
        elif ("tek adet böbrek taşı" in low) or ("a single renal calculus" in low):
            total_n = 1
        if ("çok sayıda olarak tanımlanmıştır" in low) or ("described as numerous" in low):
            qualifier = "MANY"

    for key in ("right", "left"):
        st = sorted(kid[key]["stones"], key=lambda s: (-(s["mm"] or -1), s["zone"]))
        mms = [s["mm"] for s in st if s["mm"] is not None]
        kid[key] = {
            "present": kid[key]["present"] or bool(st),
            "n_characterized": len(st),
            "stones": st,
            "max_mm": max(mms) if mms else None,
            "max_cls": (max(st, key=lambda s: s["mm"] or -1)["cls"] if mms else None),
            "zones": sorted({s["zone"] for s in st}),
        }

    n_char = kid["right"]["n_characterized"] + kid["left"]["n_characterized"]
    if qualifier == "MANY":
        total_n = None
    elif total_n is None:
        total_n = n_char if n_char else None

    sides = {s for s in ("right", "left") if kid[s]["present"]}
    laterality = ("BILATERAL" if sides == {"right", "left"}
                  else "RIGHT" if sides == {"right"}
                  else "LEFT" if sides == {"left"} else "NONE")

    all_stones = [(("RIGHT" if k == "right" else "LEFT"), s)
                  for k in ("right", "left") for s in kid[k]["stones"] if s["mm"] is not None]
    largest = max(all_stones, key=lambda x: x[1]["mm"]) if all_stones else None

    return {
        "laterality": laterality,
        "total_n": total_n,
        "total_qualifier": qualifier,
        "n_characterized": n_char,
        "kidneys": kid,
        "largest": ({"side": largest[0], "zone": largest[1]["zone"],
                     "mm": largest[1]["mm"], "cls": largest[1]["cls"]} if largest else None),
        "anomalies": sorted(anomalies),
    }


def diff_factsets(a: dict, b: dict) -> list[str]:
    """Return the differences between two fact sets in human-readable form."""
    out = []
    for f in ("laterality", "total_n", "total_qualifier", "n_characterized", "anomalies", "largest"):
        if a.get(f) != b.get(f):
            out.append(f"{f}: {a.get(f)!r} != {b.get(f)!r}")
    for k in ("right", "left"):
        ka, kb = a["kidneys"][k], b["kidneys"][k]
        for f in ("present", "n_characterized", "max_mm", "max_cls", "zones"):
            if ka.get(f) != kb.get(f):
                out.append(f"kidneys.{k}.{f}: {ka.get(f)!r} != {kb.get(f)!r}")
        if ka["stones"] != kb["stones"]:
            out.append(f"kidneys.{k}.stones: {ka['stones']!r} != {kb['stones']!r}")
    return out
