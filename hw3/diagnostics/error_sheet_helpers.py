import re

MODAL = re.compile(r"\b(should|must|need to|has to)\b", re.I)  # same as hw3_overnight.py


def modal_info(text):
    """(n_tokens, has_modal y/n, relative position of first modal or '') — same as error_sheet.py."""
    toks = text.split()
    m = MODAL.search(text)
    if not m:
        return len(toks), "n", ""
    start_tok = len(text[:m.start()].split())
    pos = start_tok / (len(toks) - 1) if len(toks) > 1 else 0.0
    return len(toks), "y", f"{pos:.2f}"
