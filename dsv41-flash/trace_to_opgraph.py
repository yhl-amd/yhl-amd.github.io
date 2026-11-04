#!/usr/bin/env python3
r"""Turn one ATOM DeepSeek-V4.1 step of a torch-profiler trace into per-(layer, op) GPU time.

The op table and its kernel-name rules live in the dsv41-flash page itself
(<script type="application/json" id="opgraph-spec">), so the page and this script
always agree on op ids. Load the JSON this writes into the page's "加载数据" box;
it stays in the browser.

Steps are picked by a CPU-side user annotation (e.g. "ATOM::forward tokens=12 ");
a kernel belongs to the step when the runtime call that launched it (matched by
correlation id, which also covers hipGraphLaunch replays) lies inside that range.
GPU-side annotation ranges are not used: they can miss the early layers.

Kernels are cut into layer halves at the spec's half markers (the mHC pre kernel
that opens every attention and FFN half, or the mono attn_pre/layer_post pair),
then each kernel is matched to an op of its half by the spec's regexes.

usage:
  trace_to_opgraph.py TRACE.json.gz --spec https://yhl-amd.github.io/dsv41-flash/ \
      --step 'ATOM::forward tokens=\d+ prefill=0' [--phase decode] [--nth 3 | --all] [-o out.json]
"""

import argparse
import gzip
import json
import re
import statistics
import sys
from collections import defaultdict


def load_spec(path):
    """The spec JSON, from the dsv41-flash page (a file path or its URL) or a .json file."""
    if path.startswith(("http://", "https://")):
        import urllib.request
        with urllib.request.urlopen(path, timeout=30) as r:
            text = r.read().decode("utf-8")
    else:
        text = open(path, encoding="utf-8").read()
    if path.endswith(".json"):
        return json.loads(text)
    m = re.search(r'<script type="application/json" id="opgraph-spec">(.*?)</script>', text, re.S)
    if not m:
        sys.exit(f"no opgraph-spec block in {path}")
    return json.loads(m.group(1))


def load_trace(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        d = json.load(f)
    return d["traceEvents"] if isinstance(d, dict) else d


def step_kernels(events, pattern):
    """[(annotation, [kernel events in GPU order])] for every matching CPU annotation."""
    anns = sorted(
        (e for e in events
         if e.get("cat") == "user_annotation" and re.search(pattern, e.get("name", ""))),
        key=lambda e: e["ts"],
    )
    runtime = defaultdict(list)  # tid -> [(ts, correlation)]
    for e in events:
        if e.get("cat") == "cuda_runtime" and "correlation" in e.get("args", {}):
            runtime[e["tid"]].append((e["ts"], e["args"]["correlation"]))
    kernels = {}
    for e in events:
        if e.get("cat") == "kernel" and "correlation" in e.get("args", {}):
            kernels.setdefault(e["args"]["correlation"], []).append(e)
    out = []
    for a in anns:
        t0, t1 = a["ts"], a["ts"] + a.get("dur", 0)
        ks = [k for ts, c in runtime[a["tid"]] if t0 <= ts <= t1 for k in kernels.get(c, ())]
        out.append((a, sorted(ks, key=lambda k: k["ts"])))
    return out


def applies(op, row, phase):
    """Whether `op` runs in `row` (a spec row dict, or a strip id) at `phase`."""
    if op.get("phases") and phase not in op["phases"]:
        return False
    if isinstance(row, str):  # strips: pro / epi / dkv / dhd
        return op["group"] == row
    rows = (op.get("rowsByPhase") or {}).get(phase) or op.get("rows")
    if rows is not None and row["id"] in rows:
        return True
    return bool(op.get("kinds")) and row["kind"] in op["kinds"]


class Matcher:
    """Attributes the kernels of one step to (row, op) with the spec's rules."""

    def __init__(self, spec, phase):
        self.spec, self.phase = spec, phase
        self.cls = {k: re.compile(v) for k, v in spec["classes"].items()}
        self.rx = {o["id"]: re.compile(o["match"]) for o in spec["ops"] if o.get("match")}
        self.half = re.compile(spec["markers"]["half"])
        self.rows = {r["id"]: r for r in spec["rows"]}
        self.ops = spec["ops"]
        self.n_layers, self.n_draft = spec["n_layers"], spec.get("n_draft", 0)
        self.counters = defaultdict(int)

    def ctx_ops(self, row_id, groups):
        row = self.rows.get(row_id)
        return [o for o in self.ops if o["group"] in groups and
                (applies(o, row, self.phase) if row else applies(o, o["group"], self.phase))]

    def match(self, row_id, groups, name, ctx_key, prev=None):
        """op dict, or None. Specific regex first, then an op that must follow `prev`
        (the op just before on this stream), then class ordinal, then bucket."""
        ops = self.ctx_ops(row_id, groups)
        for o in ops:
            if not o.get("bucket") and o["id"] in self.rx and self.rx[o["id"]].search(name):
                return o
        if prev and self.cls["fp8"].search(name):
            for o in ops:
                if o.get("after") == prev:
                    return o
        for cls, rx in self.cls.items():
            if rx.search(name):
                cands = [o for o in ops if o.get("cls") == cls]
                n = self.counters[(ctx_key, cls)]
                self.counters[(ctx_key, cls)] = n + 1
                return cands[n] if n < len(cands) else None
        for o in ops:
            if o.get("bucket") and self.rx[o["id"]].search(name):
                return o
        return None

    def anywhere(self, name, row_id):
        for o in self.ops:
            if o.get("anywhere") and self.rx[o["id"]].search(name):
                allowed = o.get("anywhereCtx")
                if not allowed or any(row_id == a or row_id.startswith(a) for a in allowed):
                    return o
        return None


def next_context(m, row_id, group):
    """Context after an FFN half's terminal op: the next row's prelude, the epilogue
    (after the last target layer), the draft head (after the last draft block)."""
    if group != "ffn":
        return None
    if row_id == f"L{m.n_layers - 1}":
        return ("tail", ("epi", "dkv"))
    if m.n_draft and row_id == f"D{m.n_draft - 1}":
        return ("dhd", ("dhd",))
    if row_id.startswith("L"):
        nxt = f"L{int(row_id[1:]) + 1}"
        if m.ctx_ops(nxt, ("pre",)):
            return (nxt, ("pre",))
    return None


def assign(spec, kernels, phase):
    """{(row, op id): [kernel, ...]} for one step's kernels, in GPU time order."""
    m = Matcher(spec, phase)
    out = defaultdict(list)
    h = -1
    ctx = ("pro", ("pro",))
    home = ctx  # the half's own context, kept to recognise a fallback switch
    pending = {}  # stream -> (quant kernel, context it ran in)
    quant_rx = m.cls["quant"]
    fp8_rx = m.cls["fp8"]

    def put(ctx_, o, ks):
        row, groups = ctx_
        if o is None:
            out[(row if row != "tail" else "epi", f"{groups[0]}.other")].extend(ks)
        else:
            out[(o["group"] if row in ("tail", "dhd", "pro") else row, o["id"])].extend(ks)

    def flush(stream):
        if stream in pending:
            qk, qctx = pending.pop(stream)
            put(qctx, m.match(qctx[0], qctx[1], qk["name"], qctx), [qk])

    ar_rx = m.cls["ar"]
    last_ar, cur = {}, -1  # half index -> index of its last "ar"-class kernel
    for i, k in enumerate(kernels):
        if m.half.search(k["name"]):
            cur += 1
        elif ar_rx.search(k["name"]):
            last_ar[cur] = i
    dispatch = next((o for o in m.ops if o["id"] == "ffn.ep_dispatch"), None)
    streams = defaultdict(int)
    for k in kernels:
        streams[k.get("args", {}).get("stream")] += 1
    main = max(streams, key=streams.get) if streams else None
    last_op = {}  # (context, stream) -> id of the op matched last there

    for i, k in enumerate(kernels):
        name = k["name"]
        stream = k.get("args", {}).get("stream")
        if m.half.search(name):
            for s in list(pending):
                flush(s)
            h += 1
            layer, part = divmod(h, 2)
            group = "attn" if part == 0 else "ffn"
            if layer < m.n_layers:
                row = f"L{layer}"
            elif layer < m.n_layers + m.n_draft:
                row = f"D{layer - m.n_layers}"
            else:
                row = "extra"
            ctx = home = (row, (group,))
            out[(row, f"{group}.mhc_pre" if "mhc" in name else
                 ("attn.mono_pre" if "attn_pre" in name else "ffn.mono_post"))].append(k)
            continue
        if ctx != home and stream != main and home[0] != "pro":
            ctx_k = home  # a side-stream kernel (shared expert) that ends after the half's collective
        else:
            ctx_k = ctx
        row, groups = ctx_k
        o = m.anywhere(name, row)
        if o:
            out[("pro", o["id"])].append(k)
            continue
        if quant_rx.search(name):
            flush(stream)
            pending[stream] = (k, ctx_k)
            continue
        carry = []
        if stream in pending:
            if fp8_rx.search(name):
                carry = [pending.pop(stream)[0]]
            else:
                flush(stream)
        if (ctx_k == home and groups[0] == "ffn" and dispatch is not None
                and ar_rx.search(name) and last_ar.get(h) != i):
            put(ctx_k, dispatch, carry + [k])  # not the half's last collective: DP token gather
            continue
        o = m.match(row, groups, name, ctx_k, last_op.get((ctx_k, stream)))
        if (o is None or o.get("bucket")) and ctx_k == home and ctx == home:
            # no terminal seen (e.g. mono fuses the all-reduce): switch when the
            # kernel clearly belongs to what follows this half
            nxt = next_context(m, row, groups[0])
            if nxt:
                o2 = m.match(nxt[0], nxt[1], name, nxt)
                if o2 is not None and not o2.get("bucket"):
                    ctx = ctx_k = nxt
                    o = o2
        put(ctx_k, o, carry + [k])
        last_op[(ctx_k, stream)] = o["id"] if o else None
        if o is not None and o.get("terminal") and ctx == home and ctx_k == home and last_ar.get(h, i) == i:
            nxt = next_context(m, row, groups[0])
            if nxt:
                ctx = nxt
    for s in list(pending):
        flush(s)
    return out


def summarize(cells):
    out = {}
    for (row, op), ks in cells.items():
        if True:
            names = defaultdict(float)
            for k in ks:
                names[re.sub(r"\(.*", "", k["name"])[:120]] += k.get("dur", 0.0)
            out[f"{row}|{op}"] = {
                "us": round(sum(k.get("dur", 0.0) for k in ks), 2),
                "n": len(ks),
                "kernels": sorted(names.items(), key=lambda kv: -kv[1])[:6],
            }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--spec", required=True, help="dsv41-flash index.html (or the spec JSON)")
    ap.add_argument("--step", required=True, help="regex on a CPU user annotation, e.g. 'ATOM::forward tokens=12 '")
    ap.add_argument("--phase", choices=["prefill", "decode"], help="defaults to the annotation's prefill=/decode= counts")
    sel = ap.add_mutually_exclusive_group()
    sel.add_argument("--nth", type=int, default=0, help="which matching step (default 0)")
    sel.add_argument("--all", action="store_true", help="median over every matching step")
    ap.add_argument("-o", "--out", default="-")
    a = ap.parse_args()

    spec = load_spec(a.spec)
    steps = step_kernels(load_trace(a.trace), a.step)
    if not steps:
        sys.exit("no annotation matches --step")
    picked = steps if a.all else [steps[a.nth]]
    name = picked[0][0]["name"]
    phase = a.phase or ("prefill" if re.search(r"prefill=[1-9]", name) else "decode")
    per_step = [summarize(assign(spec, ks, phase)) for _, ks in picked]
    ops = {}
    for key in set().union(*per_step):
        vals = [s[key]["us"] for s in per_step if key in s]
        first = next(s[key] for s in per_step if key in s)
        ops[key] = dict(first, us=round(statistics.median(vals), 2), steps=len(vals))
    halves = sum(1 for k in picked[0][1] if re.search(spec["markers"]["half"], k["name"]))
    result = {
        "opgraph": spec["version"],
        "phase": phase,
        "source": {"trace": a.trace.split("/")[-1], "step": name, "steps": len(picked),
                   "kernels": len(picked[0][1]), "halves": halves},
        "ops": ops,
    }
    expect = 2 * spec["n_layers"]
    if halves < expect:
        print(f"warning: {halves} layer halves found, expected >= {expect}; "
              "check the spec's half marker against this trace", file=sys.stderr)
    text = json.dumps(result, ensure_ascii=False, indent=1)
    if a.out == "-":
        print(text)
    else:
        open(a.out, "w", encoding="utf-8").write(text)
        print(f"wrote {a.out}: {len(ops)} (row, op) cells, {halves} halves, phase {phase}", file=sys.stderr)


if __name__ == "__main__":
    main()
