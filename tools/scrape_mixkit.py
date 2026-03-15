import argparse
import asyncio
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from scrape_core import AudioCandidate, ScrapeConfig, ScrapePipeline, print_progress


BASE = "https://mixkit.co"
SFX_HOME = "https://mixkit.co/free-sound-effects/"


def _get_soup(session: requests.Session, url: str) -> BeautifulSoup:
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def _category_pages(session: requests.Session, max_categories: int) -> list[str]:
    soup = _get_soup(session, SFX_HOME)
    out = [SFX_HOME]
    for a in soup.select("a[href^='/free-sound-effects/']"):
        href = (a.get("href") or "").strip()
        if not href or "download" in href:
            continue
        full = urljoin(BASE, href)
        if full not in out:
            out.append(full)
    return out[:max_categories]


def build_candidates(max_categories: int = 60, max_pages_per_category: int = 30) -> list[AudioCandidate]:
    out: list[AudioCandidate] = []
    seen_audio = set()
    with requests.Session() as session:
        session.headers.update({"User-Agent": "Mozilla/5.0"})
        cat_urls = _category_pages(session, max_categories)
        print(f"[mixkit] categories={len(cat_urls)}")

        for cat_url in cat_urls:
            style = cat_url.rstrip("/").split("/")[-1] or "sound-effects"
            for page_n in range(1, max_pages_per_category + 1):
                page_url = f"{cat_url}?page={page_n}" if page_n > 1 else cat_url
                try:
                    soup = _get_soup(session, page_url)
                except Exception:
                    break

                cards = soup.select(".item-grid-card, .item-grid-sfx-preview")
                if not cards:
                    break

                new_on_page = 0
                for card in cards:
                    title_el = card.select_one("h2, h3, [data-test-id*='title']")
                    title = title_el.get_text(" ", strip=True) if title_el else "mixkit_sound"

                    tags = []
                    for tag in card.select("a[href^='/free-sound-effects/']"):
                        txt = tag.get_text(" ", strip=True)
                        if txt and txt not in tags:
                            tags.append(txt)
                    tags_csv = ",".join(tags[:8])

                    btn = card.select_one("[data-download--button-modal-url-value]")
                    if not btn:
                        continue
                    modal_path = (btn.get("data-download--button-modal-url-value") or "").strip()
                    if not modal_path:
                        continue

                    audio_url = urljoin(BASE, modal_path)
                    if audio_url in seen_audio:
                        continue
                    seen_audio.add(audio_url)
                    out.append(
                        AudioCandidate(
                            source="mixkit",
                            page_url=page_url,
                            audio_url=audio_url,
                            title=title,
                            style=style,
                            tags=tags_csv,
                        )
                    )
                    new_on_page += 1

                if new_on_page == 0:
                    break
    return out


async def main() -> None:
    ap = argparse.ArgumentParser(description="Scrape Mixkit into normalized local wavs.")
    ap.add_argument("--max-categories", type=int, default=60)
    ap.add_argument("--max-pages-per-category", type=int, default=30)
    ap.add_argument("--target-count", type=int, default=1500)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--min-rms", type=float, default=0.005)
    args = ap.parse_args()

    candidates = build_candidates(
        max_categories=args.max_categories,
        max_pages_per_category=args.max_pages_per_category,
    )
    print(f"[mixkit] candidates={len(candidates)}")

    cfg = ScrapeConfig(concurrency=args.concurrency, min_rms=args.min_rms)
    pipe = ScrapePipeline(cfg)
    result = await pipe.run(candidates, target_count=args.target_count, progress_cb=print_progress)
    print(f"[mixkit] done {result}")


if __name__ == "__main__":
    asyncio.run(main())

