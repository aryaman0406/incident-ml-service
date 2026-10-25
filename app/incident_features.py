"""Feature extraction for the CI build-log line ranker.

One implementation shared by the training notebook check and the serving API, so the
deployed model sees exactly the features it was trained on. It reproduces the 'old'
feature set (52 features) of the final model. Input: raw Travis CI log text.
"""
import re
import numpy as np
import pandas as pd

ANSI_ESC = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
FOLD_RX = re.compile(r"^travis_fold:(start|end):(\S+)\s*$")
TIME_RX = re.compile(r"^travis_time:(start|end):(\w+)(?::start=(\d+),finish=(\d+),duration=(\d+))?\s*$")
PATH_LN = re.compile(r"[\w\-./]+\.\w{1,5}:\d+")
AT_FRAME = re.compile(r"\s*at\s")
CMD_EXIT = re.compile(r"^The command .* exited with \d+")

KW = ["error", "fail", "exception", "traceback", "assert", "expected", "panic", "undefined",
      "not found", "timeout", "warning", "exit"]
SEC_PREFIXES = ("system_info", "worker_info", "git", "before_install", "install", "before_script", "script",
                "after_failure", "after_script", "after_success", "after_", "announce", "apt", "cache", "test")
SEC_NAMES = ("none",) + SEC_PREFIXES + ("other",)


def clean_log(text):
    """Strip terminal codes, tag each line (content / blank / fold_marker / time_marker) and its Travis section."""
    rows, section = [], "(none)"
    for i, raw in enumerate(text.split("\n"), start=1):
        s = ANSI_ESC.sub("", raw).rstrip()
        if not s:
            kind = "blank"
        elif (m := FOLD_RX.match(s)):
            kind = "fold_marker"
            if m.group(1) == "start":
                section = m.group(2)
        elif TIME_RX.match(s):
            kind = "time_marker"
        else:
            kind = "content"
        rows.append((i, s, kind, section))
        if kind == "fold_marker" and m.group(1) == "end":
            section = "(none)"
    return rows


def sec_cat(s):
    if s == "(none)":
        return "none"
    for p in SEC_PREFIXES:
        if s.startswith(p):
            return p
    return "other"


def build_features(text):
    """Return a DataFrame with one row per content line: line_no, text, section and all features."""
    df = pd.DataFrame(clean_log(text), columns=["line_no", "text", "kind", "section"])
    c = df[df["kind"] == "content"].reset_index(drop=True)
    n = len(c)
    if n == 0:
        return c
    t = c["text"]
    idx = np.arange(n)
    lens = t.str.len().values
    c["rel_pos"] = idx / np.maximum(n - 1, 1)
    c["from_end"] = np.log1p(n - 1 - idx)
    c["log_len"] = np.log1p(n)
    c["len"] = np.log1p(lens)
    indent = np.clip(lens - t.str.lstrip().str.len().values, 0, 40)
    c["indent"] = indent
    c["digit_frac"] = t.apply(lambda s: len(re.findall(r"\d", s))).values / np.maximum(lens, 1)
    c["upper_frac"] = t.apply(lambda s: len(re.findall(r"[A-Z]", s))).values / np.maximum(lens, 1)
    has_path = t.apply(lambda s: PATH_LN.search(s) is not None).values
    is_at = t.apply(lambda s: AT_FRAME.match(s) is not None).values
    is_exit = t.apply(lambda s: CMD_EXIT.search(s) is not None).values
    c["has_path_ln"], c["is_at_frame"], c["is_cmd_exit"] = has_path, is_at, is_exit
    c["is_prompt"] = t.str.startswith("$").values
    tl = t.str.lower()
    for k in KW:
        c["kw_" + k.replace(" ", "_")] = tl.str.contains(k, regex=False).values

    def win_sum(x, k):
        cs = np.cumsum(np.asarray(x, dtype=np.int64))
        lo = np.maximum(idx - k, 0)
        hi = np.minimum(idx + k, n - 1)
        return cs[hi] - np.where(lo > 0, cs[np.maximum(lo - 1, 0)], 0)

    bases = {"kw_error": c["kw_error"].values, "kw_fail": c["kw_fail"].values,
             "kw_expected": c["kw_expected"].values, "has_path_ln": has_path, "is_at_frame": is_at}
    for name, x in bases.items():
        for k in (3, 10):
            c[f"{name}_w{k}"] = win_sum(x, k)
    for k in (3, 10):
        c[f"indented_w{k}"] = win_sum(indent > 0, k)

    ex = np.flatnonzero(is_exit)
    if len(ex) == 0:
        nxt = prv = np.full(n, np.nan)
    else:
        p1 = np.searchsorted(ex, idx, side="left")
        nxt = np.where(p1 < len(ex), ex[np.minimum(p1, len(ex) - 1)], np.nan)
        p2 = np.searchsorted(ex, idx, side="right") - 1
        prv = np.where(p2 >= 0, ex[np.maximum(p2, 0)], np.nan)
    c["d_next_exit"] = np.log1p(np.clip(np.where(np.isnan(nxt), 5000, nxt - idx), 0, 5000))
    c["d_prev_exit"] = np.log1p(np.clip(np.where(np.isnan(prv), 5000, idx - prv), 0, 5000))

    cats = c["section"].map(sec_cat)
    for name in SEC_NAMES:
        c[f"sec_{name}"] = (cats == name).values
    return c


def feature_matrix(c, feature_names):
    """Columns in the exact order the model was trained on; unseen section columns become 0."""
    return c.reindex(columns=feature_names, fill_value=0).astype(np.float32).values


def top_lines(text, model, feature_names, k=10, max_chars=300):
    """Rank the content lines of a raw log and return the k highest-scored ones."""
    c = build_features(text)
    if len(c) == 0:
        return []
    scores = model.predict_proba(feature_matrix(c, feature_names))[:, 1]
    order = np.argsort(-scores, kind="stable")[:k]
    return [{"line_no": int(c["line_no"].iloc[i]), "score": float(scores[i]),
             "section": str(c["section"].iloc[i]), "text": c["text"].iloc[i][:max_chars]} for i in order]
