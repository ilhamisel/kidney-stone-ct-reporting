"""Turkish and English report generator — a small grammar, not a format string.

The two languages are generated in parallel from the same fact set; neither is a
translation of the other.

The governing design constraint is that every fact appears EXACTLY once and in one
surface form, because `report_parser` must recover the identical fact set from the
text (the hard gate of stage 10). Variation is therefore confined to slots that
carry no clinical fact: the opening clause, the verb, the phrasing of the
measurement, and the ordering of clauses.

This replaces the earlier three-slot template, which emitted a fixed zone string
and so could not express a stone outside that zone at all.
"""
from __future__ import annotations

import hashlib
import random

from common.labels import SIZE_CLASS_EN, SIZE_CLASS_TR, ZONE_EN, ZONE_TR

# --- variant slots that carry no clinical fact ------------------------------
OPENING_TR = [
    "Kontrastsız batın BT incelemesinde",
    "Kontrastsız üriner sistem BT tetkikinde",
    "Non-kontrast abdomen BT incelemesinde",
]
VERB_TR = ["izlendi", "izlenmektedir", "saptandı", "mevcuttur"]
IMPRESSION_BIL_TR = [
    "Bilateral nefrolitiazis",
    "Her iki böbrekte nefrolitiazis",
    "İki taraflı böbrek taşı",
]
OPENING_EN = [
    "On non-contrast abdominal CT",
    "On non-contrast urinary tract CT",
    "On unenhanced abdominal CT",
]
VERB_EN = ["is seen", "is noted", "is identified", "is present"]

ANOMALY_TR = {
    "HORSESHOE_KIDNEY": "At nalı böbrek anomalisi mevcuttur; zon tanımlamaları bu anomaliye göre değerlendirilmelidir.",
    "MANY_SMALL_STONES_MIMIC_LARGE": "Çok sayıda küçük boyutlu taş mevcut olup, reformat görüntülerde tek bir büyük taş izlenimi verebilir.",
    "HYPERDENSE_PYRAMID_MIMIC": "Renal piramidlerde hiperdens görünüm mevcut olup taş ile ayırıcı tanıya girmektedir.",
    "COMPLICATED_CYST_HYPERDENSE_LESION": "Hiperdens/komplike kist ile uyumlu lezyon izlenmekte olup taş olarak değerlendirilmemiştir.",
    "URETERAL_STONE_PRESENT": "Üreterde taş dansitesi eşlik etmektedir; bu rapor böbrek taşlarını tanımlamaktadır.",
    "BLADDER_DIVERTICULUM_STONE": "Mesane divertikülü içinde taş dansitesi eşlik etmektedir.",
}
ANOMALY_EN = {
    "HORSESHOE_KIDNEY": "A horseshoe kidney anomaly is present; zonal descriptions should be interpreted accordingly.",
    "MANY_SMALL_STONES_MIMIC_LARGE": "Multiple small calculi are present and may simulate a single large calculus on reformatted images.",
    "HYPERDENSE_PYRAMID_MIMIC": "Hyperdense renal pyramids are noted and may mimic calculi.",
    "COMPLICATED_CYST_HYPERDENSE_LESION": "A hyperdense/complicated cyst is noted and was not counted as a calculus.",
    "URETERAL_STONE_PRESENT": "An accompanying ureteral calculus is present; this report describes the renal calculi.",
    "BLADDER_DIVERTICULUM_STONE": "An accompanying calculus within a bladder diverticulum is present.",
}

NUM_TR = {1: "bir", 2: "iki", 3: "üç", 4: "dört", 5: "beş"}
NUM_EN = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five"}


def make_rng(case_id: str, seed: int) -> random.Random:
    h = int(hashlib.sha1(case_id.encode("utf-8")).hexdigest()[:8], 16)
    return random.Random(seed ^ h)


def _mm_str(mm: float | None) -> str:
    if mm is None:
        return ""
    return str(int(mm)) if float(mm).is_integer() else f"{mm:g}"


def _article_en(num_str: str) -> str:
    """Choose "a" or "an" from how the number is read aloud: 8, 11, 18 and 80-89
    begin with a vowel sound."""
    n = num_str.split(".")[0]
    if n.startswith("8") or n in ("11", "18"):
        return "an"
    return "a"


# --------------------------------------------------------------------- TR
def _measure_tr(mm: float | None, variant: int) -> str:
    if mm is None:
        return "boyutu ölçülemeyen"
    s = _mm_str(mm)
    return [f"{s} mm boyutunda", f"boyutu {s} mm ölçülen", f"{s} mm çapında"][variant]


def _stone_clause_tr(stone: dict, variant: int) -> str:
    zone = ZONE_TR[stone["zone"]]
    meas = _measure_tr(stone["mm"], variant)
    cls = SIZE_CLASS_TR[stone["cls"]].lower() if stone["cls"] else "boyutu bilinmiyor"
    return f"{zone}da {meas} ({cls})"


def render_tr(fs: dict, variants: dict) -> dict:
    v_open, v_verb, v_meas, v_imp = (
        variants["opening"], variants["verb"], variants["measure"], variants["impression"]
    )
    order = variants["order"]
    findings: list[str] = []

    sides = [("right", "Sağ"), ("left", "Sol")]
    if order == 1:
        sides = sides[::-1]

    first = True
    for key, label in sides:
        k = fs["kidneys"][key]
        if not k["stones"]:
            continue
        clauses = [_stone_clause_tr(s, v_meas) for s in k["stones"]]
        body = clauses[0] if len(clauses) == 1 else ", ".join(clauses[:-1]) + " ve " + clauses[-1]
        noun = "taş dansitesi" if len(clauses) == 1 else "taş dansiteleri"
        lead = f"{OPENING_TR[v_open]} {label.lower()} böbrek" if first else f"{label} böbrek"
        findings.append(f"{lead} {body} {noun} {VERB_TR[v_verb]}.")
        first = False

    # A side known to contain a stone that was not individually characterized.
    # It must still be stated, otherwise the report would imply the kidney is clear.
    for key, label in sides:
        k = fs["kidneys"][key]
        if k["present"] and not k["stones"]:
            findings.append(f"{label} böbrekte de taş dansitesi mevcut olup ayrıca tanımlanmamıştır.")

    if not findings:
        findings.append("Her iki böbrekte taş dansitesi izlenmedi.")

    if fs["total_qualifier"] == "MANY":
        findings.append("Böbreklerde sayılamayacak kadar çok sayıda taş dansitesi mevcuttur.")
    elif fs["total_n"] is not None and fs["total_n"] > fs["n_characterized"]:
        nc = fs["n_characterized"]
        findings.append(
            f"Toplam {fs['total_n']} adet taş dansitesi izlenmiş olup, "
            f"bunlardan {nc} tanesi {'' if nc == 1 else 'ayrı ayrı '}tanımlanmıştır."
        )

    for a in fs["anomalies"]:
        if a in ANOMALY_TR:
            findings.append(ANOMALY_TR[a])

    # ---- impression
    lat = fs["laterality"]
    if lat == "NONE":
        impression = ["Böbrek taşı saptanmamıştır."]
    else:
        if lat == "BILATERAL":
            head = IMPRESSION_BIL_TR[v_imp % len(IMPRESSION_BIL_TR)]
        elif lat == "RIGHT":
            head = "Sağ böbrekte nefrolitiazis"
        else:
            head = "Sol böbrekte nefrolitiazis"
        lg = fs["largest"]
        single = fs["total_n"] == 1 and fs["n_characterized"] == 1
        if lg:
            side_tr = "sağ" if lg["side"] == "RIGHT" else "sol"
            # with a single stone, "the largest stone" is a redundant phrase
            prefix = "taş" if single else "en büyük taş"
            head += (
                f"; {prefix} {side_tr} böbrek {ZONE_TR[lg['zone']]}da "
                f"{_mm_str(lg['mm'])} mm ({SIZE_CLASS_TR[lg['cls']].lower()})"
            )
        impression = [head + "."]
        if fs["total_qualifier"] == "MANY":
            impression.append("Taş sayısı çok sayıda olarak tanımlanmıştır.")
        elif single:
            impression.append("Tek adet böbrek taşı saptanmıştır.")
        elif fs["total_n"] is not None:
            impression.append(f"Toplam {fs['total_n']} adet böbrek taşı saptanmıştır.")

    f_txt, i_txt = " ".join(findings), " ".join(impression)
    return {"findings": f_txt, "impression": i_txt,
            "full": f"BULGULAR:\n{f_txt}\n\nSONUÇ:\n{i_txt}"}


# --------------------------------------------------------------------- EN
def _stone_clause_en(stone: dict, variant: int, side: str | None = None) -> str:
    """Two structural variants, both idiomatic in radiological prose.

    The size class always follows the measurement in parentheses, which is
    conventional in structured reporting and, more importantly here, gives the
    parser a single uniform surface to match. When `side` is supplied it is
    appended to the zone phrase as "of the <side> kidney".
    """
    loc = ZONE_EN[stone["zone"]] + (f" of the {side} kidney" if side else "")
    cls = SIZE_CLASS_EN[stone["cls"]] if stone["cls"] else "of unknown size"
    if stone["mm"] is None:
        return f"a calculus of indeterminate size ({cls}) in the {loc}"
    s = _mm_str(stone["mm"])
    if variant == 1:
        return f"a calculus in the {loc} measuring {s} mm ({cls})"
    return f"{_article_en(s)} {s} mm calculus ({cls}) in the {loc}"


def render_en(fs: dict, variants: dict) -> dict:
    v_open, v_verb, v_meas = variants["opening"], variants["verb"], variants["measure"] % 2
    order = variants["order"]
    findings: list[str] = []
    sides = [("right", "right"), ("left", "left")]
    if order == 1:
        sides = sides[::-1]

    first = True
    for key, label in sides:
        k = fs["kidneys"][key]
        if not k["stones"]:
            continue
        lead = f"{OPENING_EN[v_open]}, " if first else ""
        if len(k["stones"]) == 1:
            clause = _stone_clause_en(k["stones"][0], v_meas, side=label)
            if not lead:
                clause = clause[0].upper() + clause[1:]
            findings.append(f"{lead}{clause} {VERB_EN[v_verb]}.")
        else:
            clauses = [_stone_clause_en(s, v_meas) for s in k["stones"]]
            body = ", ".join(clauses[:-1]) + " and " + clauses[-1]
            n_word = NUM_EN.get(len(clauses), str(len(clauses)))
            verb = VERB_EN[v_verb].replace("is ", "are ")
            n_word = n_word.lower() if lead else n_word
            findings.append(f"{lead}{n_word} calculi {verb} in the {label} kidney: {body}.")
        first = False

    for key, label in sides:
        k = fs["kidneys"][key]
        if k["present"] and not k["stones"]:
            findings.append(f"A calculus is also present in the {label} kidney but was not separately characterized.")

    if not findings:
        findings.append("No calculus was identified in either kidney.")

    if fs["total_qualifier"] == "MANY":
        findings.append("Numerous calculi, too many to count, are present in the kidneys.")
    elif fs["total_n"] is not None and fs["total_n"] > fs["n_characterized"]:
        nc = fs["n_characterized"]
        findings.append(
            f"A total of {fs['total_n']} calculi were reported, "
            f"of which {nc} {'is' if nc == 1 else 'are'} individually described."
        )

    for a in fs["anomalies"]:
        if a in ANOMALY_EN:
            findings.append(ANOMALY_EN[a])

    lat = fs["laterality"]
    if lat == "NONE":
        impression = ["No renal calculus identified."]
    else:
        head = {"BILATERAL": "Bilateral nephrolithiasis",
                "RIGHT": "Right-sided nephrolithiasis",
                "LEFT": "Left-sided nephrolithiasis"}[lat]
        lg = fs["largest"]
        single = fs["total_n"] == 1 and fs["n_characterized"] == 1
        if lg:
            subj = "the calculus is" if single else "the largest calculus is"
            head += (
                f"; {subj} in the {lg['side'].lower()} {ZONE_EN[lg['zone']]}, "
                f"{_mm_str(lg['mm'])} mm ({SIZE_CLASS_EN[lg['cls']]})"
            )
        impression = [head + "."]
        if fs["total_qualifier"] == "MANY":
            impression.append("The number of calculi was described as numerous.")
        elif fs["total_n"] == 1:
            impression.append("A single renal calculus was identified.")
        elif fs["total_n"] is not None:
            impression.append(f"A total of {fs['total_n']} renal calculi were identified.")

    f_txt, i_txt = " ".join(findings), " ".join(impression)
    return {"findings": f_txt, "impression": i_txt,
            "full": f"FINDINGS:\n{f_txt}\n\nIMPRESSION:\n{i_txt}"}


def choose_variants(case_id: str, seed: int) -> dict:
    rng = make_rng(case_id, seed)
    return {
        "opening": rng.randrange(3),
        "verb": rng.randrange(4),
        "measure": rng.randrange(3),
        "impression": rng.randrange(3),
        "order": rng.randrange(2),
    }
