"""Label parsing: enumerations, size classes, zone and free-text parsers.

The clinical vocabulary is Turkish, so the patterns below match Turkish source
text; the parsed output is language-independent.

There are two label sources:
  PRIMARY     the radiologist's original submission — age, sex, stone count, and
              the zone and diameter of each stone
  SECONDARY   derived from it, carrying the report text and the free-text
              laterality cues

Laterality is derived from the union of ALL stones AND the free-text cues. The
earlier pipeline derived it from the zone of the largest stone alone, which
labelled 18 bilateral patients as unilateral. Deriving it from the union makes
that error structurally impossible rather than merely unlikely.
"""
from __future__ import annotations

import re
import unicodedata

# ------------------------------------------------------------- enumerations
SIZE_CLASSES = ["MIKROLITIYAZIS", "KUCUK", "ORTA", "BUYUK", "COK_BUYUK"]
SIZE_CLASS_TR = {
    "MIKROLITIYAZIS": "Mikrolitiyazis",
    "KUCUK": "Küçük taş",
    "ORTA": "Orta boy taş",
    "BUYUK": "Büyük taş",
    "COK_BUYUK": "Çok büyük taş",
}
SIZE_CLASS_EN = {
    "MIKROLITIYAZIS": "microlithiasis",
    "KUCUK": "small",
    "ORTA": "medium-sized",
    "BUYUK": "large",
    "COK_BUYUK": "very large",
}
ZONES = ["UPPER", "MID", "LOWER", "MID_UPPER", "MID_LOWER", "UNKNOWN"]
ZONE_TR = {"UPPER": "üst zon", "MID": "orta zon", "LOWER": "alt zon",
           "MID_UPPER": "orta-üst zon", "MID_LOWER": "orta-alt zon", "UNKNOWN": "zonu belirtilmemiş"}
ZONE_EN = {"UPPER": "upper zone", "MID": "mid zone", "LOWER": "lower zone",
           "MID_UPPER": "mid-to-upper zone", "MID_LOWER": "mid-to-lower zone",
           "UNKNOWN": "unspecified zone"}
SIDE_TR = {"RIGHT": "sağ", "LEFT": "sol", "UNKNOWN": "tarafı belirtilmemiş"}
SIDE_EN = {"RIGHT": "right", "LEFT": "left", "UNKNOWN": "unspecified side"}

ANOMALIES = [
    "HORSESHOE_KIDNEY",
    "MANY_SMALL_STONES_MIMIC_LARGE",
    "HYPERDENSE_PYRAMID_MIMIC",
    "COMPLICATED_CYST_HYPERDENSE_LESION",
    "URETERAL_STONE_PRESENT",
    "BLADDER_DIVERTICULUM_STONE",
]


def size_class(mm: float | None) -> str | None:
    """Size class from a diameter in millimetres.

    The boundaries are half-open so that a decimal value falls in the intended
    class; the resulting class histogram reproduces the counts of the source
    table exactly (38 / 61 / 122 / 81 / 24).
    """
    if mm is None:
        return None
    if mm < 4:
        return "MIKROLITIYAZIS"   # <=3
    if mm < 6:
        return "KUCUK"            # 4-5
    if mm < 11:
        return "ORTA"             # 6-10
    if mm < 20:
        return "BUYUK"            # 11-19
    return "COK_BUYUK"            # >=20


def count_bucket(n: int | None, qualifier: str) -> str:
    if qualifier == "MANY" or n is None:
        return "MANY_UNSPECIFIED"
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    if n <= 4:
        return "3-4"
    if n <= 9:
        return "5-9"
    return "10+"


# ------------------------------------------------------------- text utilities
def _upper_tr(s: str) -> str:
    return str(s).replace("i", "İ").replace("ı", "I").upper()


def _ascii(s: str) -> str:
    s = _upper_tr(s)
    for a, b in [("Ç", "C"), ("Ğ", "G"), ("İ", "I"), ("Ö", "O"), ("Ş", "S"), ("Ü", "U"), ("Â", "A")]:
        s = s.replace(a, b)
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


_DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "İ": "1"})


def repair_number(tok: str) -> str:
    """Repair a numeric token damaged by transcription: O to 0, l and I to 1.

    The raw string is preserved; the repair is applied only when converting to a
    number, and the fact that it was applied is recorded. Observed case: a stone
    count written with a capital O instead of a zero.
    """
    return tok.translate(_DIGIT_FIX)


def parse_int_token(tok: str) -> int | None:
    t = repair_number(str(tok).strip())
    m = re.fullmatch(r"\d+", t)
    return int(m.group()) if m else None


# ------------------------------------------------------------- zone parsing
def parse_zone_cell(raw: str) -> dict:
    """Parse a zone cell such as "SAĞ ALT ZON" (right lower zone) into a side and
    a zone. Truncated spellings and anomaly annotations are handled explicitly."""
    out = {"side": "UNKNOWN", "zone": "UNKNOWN", "zone_inferred": False,
           "anomaly": None, "note": None}
    if raw is None:
        return out
    txt = _ascii(raw).strip()
    if not txt:
        return out

    if "AT NALI" in txt:
        out["anomaly"] = "HORSESHOE_KIDNEY"
        out["note"] = str(raw).strip()
        return out
    if "COK SAYIDA" in txt and "BUYUK TAS" in txt:
        out["anomaly"] = "MANY_SMALL_STONES_MIMIC_LARGE"
        out["note"] = str(raw).strip()
        return out
    if "HIPERDENS" in txt and "PIRAMIS" in txt:
        out["anomaly"] = "HYPERDENSE_PYRAMID_MIMIC"
        out["note"] = str(raw).strip()
        return out
    if "KIST" in txt or "LEZYON" in txt:
        out["anomaly"] = "COMPLICATED_CYST_HYPERDENSE_LESION"
        out["note"] = str(raw).strip()
        return out

    if re.search(r"\bSAG\b|\bSA\b(?!L)", txt) or txt.startswith("SAG"):
        out["side"] = "RIGHT"
    elif "SOL" in txt:
        out["side"] = "LEFT"

    # Order matters: compound zones are tested before the single ones, otherwise
    # "ORTA ALT" (mid-lower) would match the "ALT" (lower) branch first.
    if "ORTA-UST" in txt or "ORTA UST" in txt:
        out["zone"] = "MID_UPPER"
    elif "ORTA ALT" in txt or "ORTA-ALT" in txt:
        out["zone"] = "MID_LOWER"
    elif "ALT" in txt:
        out["zone"] = "LOWER"
    elif "UST" in txt:
        out["zone"] = "UPPER"
    elif "ORTA" in txt:
        out["zone"] = "MID"
    elif re.search(r"\bOR\b|\bOR$", txt):
        # A truncated spelling of "ORTA" (mid). Completed, and flagged as
        # inferred so the inference is visible downstream rather than silent.
        out["zone"] = "MID"
        out["zone_inferred"] = True
        out["note"] = str(raw).strip()

    if out["zone"] == "UNKNOWN" and out["side"] != "UNKNOWN":
        out["zone_inferred"] = True
    return out


# ---------------------------------------------------------------- serbest metin
_LATERALITY_PATTERNS = [
    (r"HER\s*IKI\s*BOBREK", "BILATERAL"),
    (r"IKI\s*TARAFLI", "BILATERAL"),
    (r"BILATERAL", "BILATERAL"),
    (r"\bSAG\s*BOBREK", "RIGHT"),
    (r"\bSOL\s*BOBREK", "LEFT"),
]


def parse_free_text(text: str) -> dict:
    """Parse a free-text finding.

    The typical pattern names a laterality, then the side and zone of the largest
    stone, its diameter, and a total count: "in both kidneys, the largest in the
    right mid zone measuring 7 mm, 10 stone densities were observed".
    """
    out = {
        "laterality_cue": None, "laterality_ambiguous": False,
        "largest_side": "UNKNOWN", "largest_zone": "UNKNOWN", "largest_mm": None,
        "declared_count": None, "count_repaired": False, "raw": text,
    }
    if not text:
        return out
    txt = _ascii(text)

    # "sol iki böbrekte" (KS0005) reads as "left, in both kidneys". It is not the
    # standard bilateral phrase and the intended meaning is a single side, so the
    # cue is taken as LEFT and the case is flagged ambiguous for adjudication.
    if re.search(r"\bSOL\s*IKI\s*BOBREK", txt):
        out["laterality_cue"] = "LEFT"
        out["laterality_ambiguous"] = True
    else:
        for pat, val in _LATERALITY_PATTERNS:
            if re.search(pat, txt):
                out["laterality_cue"] = val
                break

    # Side and zone of the largest stone: "büyüğü <side> <zone> zonda".
    m = re.search(r"BUYUGU\s+(SAG|SOL)?\s*(ALT|ORTA|UST)\s*ZON", txt)
    if m:
        if m.group(1):
            out["largest_side"] = "RIGHT" if m.group(1) == "SAG" else "LEFT"
        elif out["laterality_cue"] in ("RIGHT", "LEFT"):
            # "in the right kidney, the largest in the mid zone" — the side is
            # stated at the start of the sentence, not next to the zone.
            out["largest_side"] = out["laterality_cue"]
        out["largest_zone"] = {"ALT": "LOWER", "ORTA": "MID", "UST": "UPPER"}[m.group(2)]

    m = re.search(r"(\d+(?:[.,]\d+)?)\s*MM", txt)
    if m:
        out["largest_mm"] = float(m.group(1).replace(",", "."))

    # The declared count, with the transcription repair applied.
    m = re.search(r"([0-9OlI]+)\s*ADET", txt)
    if m:
        raw_tok = m.group(1)
        n = parse_int_token(raw_tok)
        if n is not None:
            out["declared_count"] = n
            out["count_repaired"] = raw_tok != repair_number(raw_tok)
    return out


# ------------------------------------------------------- structured findings
_LESION_RE = re.compile(r"^(?P<loc>.*?)\s+yerleşimli,\s*(?P<size>.*?)\s*\((?P<cls>.*?)\)$")


def parse_structured_findings(text: str) -> list[dict]:
    """Parse a semicolon-separated structured finding — location, diameter and
    size class per lesion — into a list of lesions."""
    out = []
    if not text:
        return out
    for part in str(text).split(";"):
        part = part.strip()
        if not part:
            continue
        m = _LESION_RE.match(part)
        if not m:
            out.append({"raw": part, "parsed": False})
            continue
        size_txt = m.group("size").strip()
        mm = None
        ms = re.match(r"(\d+(?:[.,]\d+)?)\s*mm", size_txt)
        if ms:
            mm = float(ms.group(1).replace(",", "."))
        z = parse_zone_cell(m.group("loc"))
        out.append({
            "raw": part, "parsed": True,
            "zone_raw": m.group("loc").strip(),
            "side": z["side"], "zone": z["zone"], "zone_inferred": z["zone_inferred"],
            "anomaly": z["anomaly"], "note": z["note"],
            "size_mm": mm, "size_text": size_txt,
            "size_class_text": m.group("cls").strip(),
        })
    return out


def strip_patient_name(text: str | None, case_id: str) -> str | None:
    """Replace the patient-name line of the original report text with the
    pseudonymous case identifier.

    The full report field carries the real patient name. Because that field enters
    the publishable subtree, it must be cleaned at the source rather than at export
    time; a QA check asserts that no name survives.
    """
    if not text:
        return text
    return re.sub(r"(?m)^\s*Hasta\s*:\s*.*$", f"Hasta: {case_id}", str(text))


def derive_laterality(stones: list[dict], free_cue: str | None) -> str:
    """Laterality is the union of the sides of all stones together with the
    free-text cue — never the side of the largest stone alone."""
    sides = {s.get("side") for s in stones if s.get("side") in ("RIGHT", "LEFT")}
    if free_cue == "BILATERAL":
        return "BILATERAL"
    if free_cue in ("RIGHT", "LEFT"):
        sides.add(free_cue)
    if sides == {"RIGHT", "LEFT"}:
        return "BILATERAL"
    if sides == {"RIGHT"}:
        return "RIGHT"
    if sides == {"LEFT"}:
        return "LEFT"
    return "NONE"
