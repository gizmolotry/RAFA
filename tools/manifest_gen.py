#!/usr/bin/env python3
"""
RAFA - Manifest Generator (works)

Zero-pandas, fast, and robust for VGGSound-style datasets.
- Recursively indexes your audio_dir ONCE (extensions configurable)
- Extracts YouTube IDs from filenames (start, middle, or end)
- Matches rows by ID + nearest start time from filename (e.g., <id>_30_40.ext)
- Streams the CSV (no huge memory), auto-detects delimiter, header optional
- Lets you specify columns by NAME or INDEX (0-based), or uses smart defaults
- Writes train/val[/test] CSVs with: wav_path,start_seconds,end_seconds,label,youtube_id

Examples (Windows):
python manifest_gen.py --audio_dir "C:/.../VGGSound/audio" --meta_csv "C:/.../vggsound_header.csv" \
  --out_train manifest_train.csv --out_val manifest_val.csv --val_ratio 0.1 \
  --exts .wav,.m4a,.mp3,.flac,.ogg,.opus,.webm,.mp4

If your meta has no header:
python manifest_gen.py --audio_dir ".../audio" --meta_csv ".../vggsound.csv" \
  --yt_col 0 --start_col 1 --label_col 2 --split_col 3
"""
import argparse
import csv
import os
import re
import sys
import random
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple, Union

# ------------------------------
# Utilities
# ------------------------------

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def sniff_csv(path: str, sample_bytes: int = 256 * 1024) -> Tuple[csv.Dialect, bool]:
    """Return (dialect, has_header) using csv.Sniffer. Fallbacks to excel dialect and header=True."""
    try:
        with open(path, 'r', encoding='utf-8-sig', newline='') as f:
            sample = f.read(sample_bytes)
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample)
        has_header = sniffer.has_header(sample)
        return dialect, has_header
    except Exception:
        # Fallback
        return csv.excel(), True


def to_int(s: Union[str, int, None]) -> Optional[int]:
    if s is None:
        return None
    if isinstance(s, int):
        return s
    s = str(s).strip()
    if s.isdigit() or (s.startswith('-') and s[1:].isdigit()):
        try:
            return int(s)
        except Exception:
            return None
    return None


# ------------------------------
# Filename parsing & index
# ------------------------------

YTID_RE_STRICT = re.compile(r'^[A-Za-z0-9_-]{11}$')
# Find any 11-char YT-like token anywhere in the string
YTID_RE_ANY = re.compile(r'([A-Za-z0-9_-]{11})')

# Extract start/end numbers that appear AFTER the first occurrence of the ID
NUMS_AFTER_ID = re.compile(r'_(\d+)(?:_(\d+))?(?=[^0-9]*$)')  # prefers last "_NN[_MM]" before extension
ALL_INTS = re.compile(r'(\d+)')


def extract_ytid_from_name(name: str, id_regex: Optional[str] = None) -> Optional[str]:
    base = os.path.basename(name)
    stem, _ = os.path.splitext(base)

    # 1) custom regex if provided
    if id_regex:
        m = re.search(id_regex, stem)
        if m:
            return m.group(1) if m.groups() else m.group(0)

    # 2) starts with ID
    first_token = stem.split('_', 1)[0]
    if YTID_RE_STRICT.match(first_token):
        return first_token

    # 3) any 11-char token anywhere
    m = YTID_RE_ANY.search(stem)
    if m and YTID_RE_STRICT.match(m.group(1)):
        return m.group(1)

    return None


def extract_start_end_from_name(name: str, ytid: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    base = os.path.basename(name)
    stem, _ = os.path.splitext(base)

    after = stem
    if ytid and ytid in stem:
        after = stem.split(ytid, 1)[1]

    m = NUMS_AFTER_ID.search(after)
    if m:
        s = float(m.group(1)) if m.group(1) is not None else None
        e = float(m.group(2)) if m.group(2) is not None else None
        return s, e

    # fallback: grab last two integers in the stem
    ints = [int(x) for x in ALL_INTS.findall(stem)]
    if ints:
        s = float(ints[-1])
        e = float(ints[-2]) if len(ints) >= 2 else None
        return s, e

    return None, None


class AudioIndex:
    def __init__(self, audio_dir: str, exts: Iterable[str]):
        self.audio_dir = audio_dir
        # Normalize extensions; support '*' or 'all' to mean any file
        norm = []
        for e in exts:
            e = (e or '').strip().lower()
            if not e:
                continue
            if e in ('*', 'all'):
                norm.append('*')
            else:
                norm.append(e if e.startswith('.') else f'.{e}')
        self.exts = tuple(norm)
        self.map: Dict[str, List[str]] = defaultdict(list)
        self.map_lower: Dict[str, List[str]] = defaultdict(list)

    def build(self) -> None:
        root = self.audio_dir
        if not os.path.isdir(root):
            raise FileNotFoundError(f"audio_dir not found: {root}")
        print(f"[manifest] Indexing audio under: {root} (recursive)")
        nfiles = 0
        for r, _, files in os.walk(root):
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                if (not self.exts) or ('*' in self.exts) or (ext in self.exts):
                    full = os.path.join(r, fn)
                    ytid = extract_ytid_from_name(fn)
                    if not ytid:
                        continue
                    self.map[ytid].append(full)
                    self.map_lower[ytid.lower()].append(full)
                    nfiles += 1
        print(f"[manifest] Indexed {nfiles} files across {len(self.map)} youtube_ids")

    @staticmethod
    def _score(path: str, ytid: str, target_start: Optional[float]) -> Tuple[float, float]:
        s, _ = extract_start_end_from_name(path, ytid)
        if target_start is None or s is None:
            return (1e9, 0.0)  # neutral order
        return (abs(s - target_start), s)

    def find(self, ytid: str, target_start: Optional[float]) -> Optional[str]:
        cand = self.map.get(ytid) or self.map_lower.get(ytid.lower())
        if not cand:
            return None
        # prefer nearest start time if possible
        best = min(cand, key=lambda p: self._score(p, ytid, target_start))
        return best


# ------------------------------
# CSV row handling
# ------------------------------

CAND_YT = ["youtube_id", "ytid", "yt", "video_id", "id"]
CAND_START = ["start_seconds", "start", "t", "offset"]
CAND_END = ["end_seconds", "end", "t_end", "offset_end"]
CAND_LABEL = ["label", "class", "category", "caption", "text"]
CAND_SPLIT = ["split", "set", "partition"]


def resolve_colnames(headers: List[str], yt_col, start_col, end_col, label_col, split_col) -> Tuple[int, Optional[int], Optional[int], Optional[int], Optional[int]]:
    """Return indices (yt_idx, start_idx, end_idx, label_idx, split_idx). Uses names or indices or defaults."""
    n = len(headers)

    def pick(spec, candidates, default_idx=None) -> Optional[int]:
        idx = to_int(spec)
        if idx is not None:
            if 0 <= idx < n:
                return idx
            raise ValueError(f"Column index {idx} out of range for headers {headers}")
        if spec is not None:
            name = str(spec)
            if name in headers:
                return headers.index(name)
            raise ValueError(f"Column '{name}' not found in CSV headers: {headers}")
        for c in candidates:
            if c in headers:
                return headers.index(c)
        return default_idx

    yt_idx    = pick(yt_col,    CAND_YT,    0)
    start_idx = pick(start_col, CAND_START, 1 if n > 1 else None)
    end_idx   = pick(end_col,   CAND_END,   None)
    label_idx = pick(label_col, CAND_LABEL, 2 if n > 2 else None)
    split_idx = pick(split_col, CAND_SPLIT, 3 if n > 3 else None)

    if yt_idx is None:
        raise ValueError("Could not resolve YouTube ID column. Pass --yt_col or add a 'youtube_id' header.")
    return yt_idx, start_idx, end_idx, label_idx, split_idx


def normalize_split(tag: Optional[str]) -> str:
    t = (tag or '').strip().lower()
    if t in ("train", "tr", "training"): return "train"
    if t in ("val", "valid", "validation", "dev"): return "val"
    if t in ("test", "testing", "eval"): return "test"
    return "train"  # default


def write_local_manifest(args) -> None:
    """Build manifests directly from a local metadata CSV with wav paths."""
    headers_out = ['wav_path', 'start_seconds', 'end_seconds', 'label', 'youtube_id', 'source', 'style']
    f_train = open(args.out_train, 'w', newline='', encoding='utf-8')
    w_train = csv.writer(f_train); w_train.writerow(headers_out)
    f_val = open(args.out_val, 'w', newline='', encoding='utf-8')
    w_val = csv.writer(f_val); w_val.writerow(headers_out)
    w_test = None; f_test = None
    if args.out_test:
        f_test = open(args.out_test, 'w', newline='', encoding='utf-8')
        w_test = csv.writer(f_test); w_test.writerow(headers_out)

    dialect, has_header = sniff_csv(args.local_meta_csv)
    rng = random.Random(args.seed)
    n_total = n_write = n_skip = 0

    with open(args.local_meta_csv, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f, dialect=dialect) if has_header else None
        if reader is None:
            raise SystemExit("--local_meta_csv requires a header row with at least wav_path.")
        headers = [h.strip() for h in (reader.fieldnames or [])]
        if args.path_col not in headers:
            raise SystemExit(f"path column '{args.path_col}' not found in {args.local_meta_csv}")

        for row in reader:
            n_total += 1
            wav_path = str(row.get(args.path_col, '')).strip()
            if not wav_path or not os.path.isfile(wav_path):
                n_skip += 1
                continue

            start_s = ''
            end_s = ''
            if args.duration_col and row.get(args.duration_col):
                try:
                    dur = float(row.get(args.duration_col))
                    end_s = max(0.0, min(dur, 10.0))
                except Exception:
                    end_s = ''

            source = str(row.get(args.source_col, '')).strip() if args.source_col else ''
            style = str(row.get(args.style_col, '')).strip() if args.style_col else ''
            label = style or source or "unknown"

            if args.split_col and row.get(args.split_col):
                bucket = normalize_split(row.get(args.split_col))
            else:
                r = rng.random()
                if args.test_ratio > 0 and r < args.test_ratio:
                    bucket = 'test'
                elif r < args.test_ratio + args.val_ratio:
                    bucket = 'val'
                else:
                    bucket = 'train'

            out_row = [wav_path, start_s, end_s, label, '', source, style]
            if bucket == 'train':
                w_train.writerow(out_row)
            elif bucket == 'val':
                w_val.writerow(out_row)
            else:
                if w_test is None:
                    w_val.writerow(out_row)
                else:
                    w_test.writerow(out_row)
            n_write += 1

    f_train.close(); f_val.close()
    if f_test: f_test.close()
    print(f"[manifest-local] DONE. total={n_total} written={n_write} skipped={n_skip}")


# ------------------------------
# Main
# ------------------------------

def main():
    ap = argparse.ArgumentParser(description="Fast robust manifest generator for VGGSound-like data")
    ap.add_argument('--audio_dir', default=None)
    ap.add_argument('--meta_csv', default=None)
    ap.add_argument('--local_meta_csv', default=None, help='Optional local metadata CSV with wav_path rows (bypasses YouTube-ID matching)')
    ap.add_argument('--out_train', default='manifest_train.csv')
    ap.add_argument('--out_val',   default='manifest_val.csv')
    ap.add_argument('--out_test',  default=None)
    ap.add_argument('--val_ratio', type=float, default=0.1, help='Used only if no split column present')
    ap.add_argument('--test_ratio', type=float, default=0.0, help='Used only if no split column present')

    # Column specs can be names or indices
    ap.add_argument('--yt_col')
    ap.add_argument('--start_col')
    ap.add_argument('--end_col')
    ap.add_argument('--label_col')
    ap.add_argument('--split_col')
    ap.add_argument('--path_col', default='wav_path')
    ap.add_argument('--style_col', default='style')
    ap.add_argument('--source_col', default='source')
    ap.add_argument('--duration_col', default='duration_sec')

    ap.add_argument('--exts', default='.wav,.m4a,.mp3,.flac,.ogg,.opus,.webm,.mp4')
    ap.add_argument('--id_regex', default=None, help='Optional custom regex with a capturing group for YouTube ID')
    ap.add_argument('--seed', type=int, default=1337)

    args = ap.parse_args()

    if args.local_meta_csv:
        write_local_manifest(args)
        return
    if not args.audio_dir or not args.meta_csv:
        raise SystemExit("Either --local_meta_csv OR both --audio_dir and --meta_csv are required.")

    # Build audio index
    exts = [e.strip() for e in args.exts.split(',') if e.strip()]
    index = AudioIndex(args.audio_dir, exts)
    index.build()

    # Prepare writers
    headers_out = ['wav_path', 'start_seconds', 'end_seconds', 'label', 'youtube_id']
    f_train = open(args.out_train, 'w', newline='', encoding='utf-8')
    w_train = csv.writer(f_train); w_train.writerow(headers_out)
    f_val   = open(args.out_val,   'w', newline='', encoding='utf-8')
    w_val   = csv.writer(f_val);   w_val.writerow(headers_out)
    w_test = None; f_test = None
    if args.out_test:
        f_test = open(args.out_test, 'w', newline='', encoding='utf-8')
        w_test = csv.writer(f_test); w_test.writerow(headers_out)

    # Read meta CSV streaming
    dialect, has_header = sniff_csv(args.meta_csv)
    with open(args.meta_csv, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f, dialect)
        first = next(reader, None)
        if first is None:
            raise SystemExit('meta_csv is empty')

        # Determine headers
        if has_header:
            headers = first
        else:
            # If first looks like headers (non-numeric, not an 11-char ID), still treat as headers
            looks_header = not YTID_RE_STRICT.match(first[0])
            headers = first if looks_header else [f'col{i}' for i in range(len(first))]
            if not looks_header:
                # Process the first row as data later
                reader = (row for row in ([first] + list(reader)))

        yt_idx, start_idx, end_idx, label_idx, split_idx = resolve_colnames(headers, args.yt_col, args.start_col, args.end_col, args.label_col, args.split_col)

        rng = random.Random(args.seed)
        n_total = n_write = n_skip = 0

        def write_row(path: str, s: Optional[float], e: Optional[float], lab: str, ytid: str, bucket: str):
            nonlocal n_write
            row = [path, '' if s is None else s, '' if e is None else e, lab, ytid]
            if bucket == 'train':
                w_train.writerow(row)
            elif bucket == 'val':
                w_val.writerow(row)
            else:
                if w_test is None:
                    # If no test file requested, fold test into val
                    w_val.writerow(row)
                else:
                    w_test.writerow(row)
            n_write += 1

        # Iterate rows
        for row in reader:
            n_total += 1
            # pad short rows
            if len(row) < len(headers):
                row = row + [''] * (len(headers) - len(row))

            ytid = str(row[yt_idx]).strip()
            if not ytid:
                n_skip += 1
                continue

            # start/end/label
            s = None
            if start_idx is not None and row[start_idx] != '':
                try: s = float(row[start_idx])
                except Exception: s = None
            e = None
            if end_idx is not None and row[end_idx] != '':
                try: e = float(row[end_idx])
                except Exception: e = None
            lab = ''
            if label_idx is not None:
                lab = str(row[label_idx]).strip()

            # find file
            path = index.find(ytid, s)
            if not path:
                n_skip += 1
                if n_skip <= 10:
                    eprint(f"[manifest] WARN: no file for {ytid} @ {s}")
                continue

            # if s/e missing in CSV, try to infer from filename
            fs, fe = extract_start_end_from_name(path, ytid)
            if s is None: s = fs
            if e is None: e = fe

            # split bucket determination
            bucket = 'train'
            if split_idx is not None:
                bucket = normalize_split(row[split_idx])
            else:
                # ratio split on the fly
                r = rng.random()
                if args.test_ratio > 0 and r < args.test_ratio:
                    bucket = 'test'
                elif r < args.test_ratio + args.val_ratio:
                    bucket = 'val'
                else:
                    bucket = 'train'

            write_row(path, s, e, lab, ytid, bucket)

            if n_total % 5000 == 0:
                print(f"[manifest] processed={n_total} written={n_write} skipped={n_skip}")

    # Close files
    f_train.close(); f_val.close();
    if f_test: f_test.close()

    print(f"[manifest] DONE. total={n_total} written={n_write} skipped={n_skip}")
    if n_write == 0:
        eprint("[manifest] No rows written. Check that your filenames contain 11-char YouTube IDs and that --exts covers your audio formats.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        eprint("Interrupted.")
        sys.exit(130)
