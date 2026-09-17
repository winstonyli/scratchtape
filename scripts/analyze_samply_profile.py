"""Summarizes a samply profile by self/inclusive sample count per function.

Usage (from repo root):
    cargo build --profile profiling --example <name>
    samply record --save-only --unstable-presymbolicate \
        -o samply_profile.json.gz -- ./target/profiling/examples/<name>.exe
    python3 scripts/analyze_samply_profile.py

Exists because the hosted Firefox Profiler UI can't import local profiles
from a WebKit-based browser (blocked by a Safari-specific limitation) -
this parses the raw profile + symbol sidecar directly instead of relying
on that UI.
"""
import gzip, json, bisect
from collections import Counter

with gzip.open('samply_profile.json.gz', 'rt') as f:
    data = json.load(f)
with open('samply_profile.json.syms.json') as f:
    syms = json.load(f)

libs = data['libs']
sym_string_table = syms['string_table']
sym_data = syms['data']  # list aligned with libs, one entry per lib (or None if no symbols)

# Build per-lib sorted (rva, size, name) lists for fast lookup.
lib_symbols = []
for i, lib in enumerate(libs):
    entry = sym_data[i] if i < len(sym_data) else None
    if not entry or not entry.get('symbol_table'):
        lib_symbols.append(None)
        continue
    st = sorted(entry['symbol_table'], key=lambda s: s['rva'])
    rvas = [s['rva'] for s in st]
    names = [sym_string_table[s['symbol']] for s in st]
    lib_symbols.append((rvas, names, lib['name']))

def resolve(lib_idx, address):
    if lib_idx is None or address is None or address < 0:
        return None
    entry = lib_symbols[lib_idx] if lib_idx < len(lib_symbols) else None
    if entry is None:
        return None
    rvas, names, libname = entry
    i = bisect.bisect_right(rvas, address) - 1
    if i < 0:
        return None
    return f"{names[i]} [{libname}]"

t = data['threads'][0]
func_resource = t['funcTable']['resource']  # index into resourceTable, per func (or -1)
func_name_idx = t['funcTable']['name']
resource_lib = t['resourceTable']['lib']  # index into libs, per resource
strings = t['stringArray']

frame_func = t['frameTable']['func']
frame_address = t['frameTable']['address']
stack_frame = t['stackTable']['frame']
stack_prefix = t['stackTable']['prefix']
sample_stack = t['samples']['stack']

# Resolve every frame once, cache by frame index.
frame_cache = {}
def resolve_frame(frame_idx):
    if frame_idx in frame_cache:
        return frame_cache[frame_idx]
    fn = frame_func[frame_idx]
    res = func_resource[fn]
    addr = frame_address[frame_idx]
    if res is None or res < 0:
        name = strings[func_name_idx[fn]]  # JS/pseudo name, use as-is
    else:
        lib_idx = resource_lib[res]
        resolved = resolve(lib_idx, addr)
        name = resolved if resolved else strings[func_name_idx[fn]]
    frame_cache[frame_idx] = name
    return name

self_counts = Counter()
inclusive_counts = Counter()

for stack_idx in sample_stack:
    if stack_idx is None:
        continue
    leaf_frame = stack_frame[stack_idx]
    self_counts[resolve_frame(leaf_frame)] += 1

    seen = set()
    cur = stack_idx
    while cur is not None:
        frame = stack_frame[cur]
        name = resolve_frame(frame)
        if name not in seen:
            seen.add(name)
            inclusive_counts[name] += 1
        cur = stack_prefix[cur]

total_samples = sum(self_counts.values())
print(f"total samples (main thread): {total_samples} (~{total_samples/1000:.1f}s at 1000Hz)\n")

print("=== TOP 30 by SELF time ===")
for name, count in self_counts.most_common(30):
    print(f"  {count:>6} ({100*count/total_samples:5.1f}%)  {name}")

print("\n=== TOP 40 by INCLUSIVE time, filtered to tiny_lm.exe/engine symbols ===")
filtered = {k: v for k, v in inclusive_counts.items() if 'tiny_lm' in k}
for name, count in Counter(filtered).most_common(40):
    print(f"  {count:>6} ({100*count/total_samples:5.1f}%)  {name}")
