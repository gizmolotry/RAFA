import argparse
import asyncio
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from scrape_core import AudioCandidate, ScrapeConfig, ScrapePipeline, print_progress


BASE = "https://soundbible.com/"


def _get_soup(session: requests.Session, url: str) -> BeautifulSoup:
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def build_candidates(start_page: int = 1, max_pages: int = 300) -> list[AudioCandidate]:
    out: list[AudioCandidate] = []
    seen_audio = set()
    with requests.Session() as session:
        session.headers.update({"User-Agent": "Mozilla/5.0"})
        for page_num in range(start_page, start_page + max_pages):
            list_url = urljoin(BASE, f"free-sound-effects-{page_num}.html")
            try:
                soup = _get_soup(session, list_url)
            except Exception:
                continue

            item_links = []
            for a in soup.select(".course-content h3 a[href], .post h3 a[href], h3 a[href]"):
                href = (a.get("href") or "").strip()
                if href.endswith(".html"):
                    item_links.append(urljoin(BASE, href))
            if not item_links and page_num > start_page + 10:
                break

            for item_url in item_links:
                try:
                    ss = _get_soup(session, item_url)
                except Exception:
                    continue

                title_tag = ss.select_one("h1, h3")
                title = title_tag.get_text(" ", strip=True) if title_tag else "soundbible_clip"

                tags = []
                for a in ss.select("a[href*='tags-']"):
                    txt = a.get_text(" ", strip=True)
                    if txt and txt not in tags:
                        tags.append(txt)
                style = tags[0] if tags else "unknown"
                tags_csv = ",".join(tags[:10])

                for a in ss.select("a[href*='grab.php']"):
                    href = (a.get("href") or "").strip()
                    if "type=wav" not in href.lower() and "type=mp3" not in href.lower():
                        continue
                    audio_url = urljoin(item_url, href)
                    if audio_url in seen_audio:
                        continue
                    seen_audio.add(audio_url)
                    out.append(
                        AudioCandidate(
                            source="soundbible",
                            page_url=item_url,
                            audio_url=audio_url,
                            title=title,
                            style=style,
                            tags=tags_csv,
                        )
                    )
    return out


async def main() -> None:
    ap = argparse.ArgumentParser(description="Scrape SoundBible into normalized local wavs.")
    ap.add_argument("--start-page", type=int, default=1)
    ap.add_argument("--max-pages", type=int, default=300)
    ap.add_argument("--target-count", type=int, default=2500)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--min-rms", type=float, default=0.005)
    args = ap.parse_args()

    candidates = build_candidates(start_page=args.start_page, max_pages=args.max_pages)
    print(f"[soundbible] candidates={len(candidates)}")

    cfg = ScrapeConfig(concurrency=args.concurrency, min_rms=args.min_rms)
    pipe = ScrapePipeline(cfg)
    result = await pipe.run(candidates, target_count=args.target_count, progress_cb=print_progress)
    print(f"[soundbible] done {result}")


if __name__ == "__main__":
    asyncio.run(main())

