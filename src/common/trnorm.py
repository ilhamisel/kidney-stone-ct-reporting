"""Turkish name normalization and the hand-verified alias table.

Matching a source folder name to a label-table row proceeds in three layers:
  1. ASCII folding with a Turkish letter map      -> EXACT / ASCII_FOLDED
  2. removal of diagnostic suffixes               -> SUFFIX_STRIPPED
  3. the hand-verified alias table                -> ALIAS

There is NO automatic acceptance by fuzzy string distance. Every exception is
recorded explicitly, because a fuzzy match that silently pairs two different
patients is the one error this stage must not be able to make.
"""
from __future__ import annotations

import re
from pathlib import Path

# Turkish letters folded to ASCII. Note that both dotted and dotless i fold to
# "I": the distinction is meaningful in Turkish but is not preserved consistently
# across the source spreadsheets, so it cannot be used for matching.
_TR_MAP = str.maketrans(
    {
        "Ç": "C", "ç": "C",
        "Ğ": "G", "ğ": "G",
        "İ": "I", "ı": "I", "i": "I",
        "Ö": "O", "ö": "O",
        "Ş": "S", "ş": "S",
        "Ü": "U", "ü": "U",
        "Â": "A", "â": "A",
        "Î": "I", "î": "I",
        "Û": "U", "û": "U",
    }
)

# Diagnostic notes appended to source folder names. These are not part of the
# name and the match fails if they are left in place. Applied in order.
_SUFFIX_PATTERNS = [
    r"\bBB\s*\+?\s*URETER.*$",
    r"\bB\s*\+\s*URETER.*$",
    r"\+\s*URETER.*$",
    r"\+\s*HIPERDENS.*$",
    r"\bAT\s*NALI\s*BOBREK.*$",
    r"\bHIPERDENS\s*PIRAMIS.*$",
    r"\bMESANE.*$",
    r"-\d+$",  # second-examination folders, named "<name>-2"
]

# Hand-verified spelling differences, mapping the name in the label table to the
# name of the source folder, both raw. There are seven in this cohort and all are
# letter-level: a transposition, a missing letter, a name written without a space.
# Each was inspected individually.
#
# The table consists of patient names and is therefore NOT part of the repository;
# it is read from the private directory (see docs/data-availability.md). Schema:
# {"label-table name": "folder name"}. If the file is absent the alias layer is
# simply disabled and unmatched names are reported as MANUAL by stage 02 — they
# are never dropped silently.
def _load_aliases() -> dict[str, str]:
    import json
    import os

    root = Path(os.environ.get("KIDNEYCT_ROOT", Path(__file__).resolve().parents[2] / "workdir"))
    f = root / "00_private" / "name_aliases.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


ALIASES: dict[str, str] = _load_aliases()


def fold(name: str) -> str:
    """Fold Turkish letters to ASCII and discard everything that is not a letter.

    Whitespace is discarded too, so that a name written without spaces folds to the
    same key as the spaced form.
    """
    s = str(name).upper().strip().translate(_TR_MAP)
    return re.sub(r"[^A-Z]", "", s)


def strip_suffix(name: str) -> str:
    """Strip the diagnostic suffixes, then fold."""
    s = str(name).upper().strip().translate(_TR_MAP)
    s = re.sub(r"\s+", " ", s)
    for pat in _SUFFIX_PATTERNS:
        s = re.sub(pat, "", s).strip()
    return re.sub(r"[^A-Z]", "", s)


def name_key(name: str) -> str:
    """The matching key: alias substitution, then suffix stripping, then folding."""
    raw = str(name).strip()
    if raw in ALIASES:
        raw = ALIASES[raw]
    return strip_suffix(raw)


def match_method(excel_name: str, folder_name: str) -> str:
    """Report which layer matched the two names, for the audit trail."""
    if excel_name.strip() == folder_name.strip():
        return "EXACT"
    if excel_name.strip() in ALIASES:
        return "ALIAS"
    if fold(excel_name) == fold(folder_name):
        return "ASCII_FOLDED"
    if strip_suffix(excel_name) == strip_suffix(folder_name):
        return "SUFFIX_STRIPPED"
    return "MANUAL"
