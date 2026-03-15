import argparse
import csv
import subprocess
import sys
from collections import Counter
from pathlib import Path


def count_wavs(root: Path) -> int:
    return sum(1 for _ in root.rglob("*.wav"))


def run_cmd(args: list[str]) -> int:
    print(f"[high-volume] run: {' '.join(args)}")
    cp = subprocess.run(args)
    return cp.returncode


def summarize_metadata(csv_path: Path) -> tuple[Counter, Counter]:
    by_source = Counter()
    by_style = Counter()
    if not csv_path.exists():
        return by_source, by_style
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            by_source[(row.get("source") or "unknown").strip() or "unknown"] += 1
            by_style[(row.get("style") or "unknown").strip() or "unknown"] += 1
    return by_source, by_style


def main() -> int:
    ap = argparse.ArgumentParser(description="Run all scrapers in high-volume mode.")
    ap.add_argument("--target-total", type=int, default=5000)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    wav_dir = Path("wav_files")
    wav_dir.mkdir(parents=True, exist_ok=True)

    start_count = count_wavs(wav_dir)
    print(f"[high-volume] start wav count={start_count}")

    jobs = [
        [
            args.python,
            "tools/scrape_wavsource.py",
            "--max-pages",
            "5000",
            "--target-count",
            "3200",
            "--concurrency",
            "28",
        ],
        [
            args.python,
            "tools/scrape_soundbible.py",
            "--start-page",
            "1",
            "--max-pages",
            "450",
            "--target-count",
            "2200",
            "--concurrency",
            "28",
        ],
        [
            args.python,
            "tools/scrape_freewavesamples.py",
            "--max-pages",
            "500",
            "--target-count",
            "1700",
            "--concurrency",
            "28",
        ],
        [
            args.python,
            "tools/scrape_mixkit.py",
            "--max-categories",
            "80",
            "--max-pages-per-category",
            "35",
            "--target-count",
            "1300",
            "--concurrency",
            "20",
        ],
    ]

    for cmd in jobs:
        code = run_cmd(cmd)
        if code != 0:
            print(f"[high-volume] warning: command failed with code={code}")
        now = count_wavs(wav_dir)
        print(f"[high-volume] wav count now={now}")
        if now >= args.target_total:
            break

    final_count = count_wavs(wav_dir)
    print(f"[high-volume] final wav count={final_count}")

    meta_csv = Path("logs/scrape_metadata.csv")
    by_source, by_style = summarize_metadata(meta_csv)
    print("[high-volume] distribution by source:")
    for k, v in by_source.most_common():
        print(f"  {k}: {v}")
    print("[high-volume] distribution by style (top 25):")
    for k, v in by_style.most_common(25):
        print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

