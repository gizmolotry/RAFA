
import asyncio
from playwright.async_api import async_playwright
import os
import httpx
import re
import time

def sanitize_filename(filename):
    """
    Removes invalid characters from a filename.
    """
    return re.sub(r'[\\/*?:"<>|]',"", filename)

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto("https://soundbible.com/tags-wave.html")

        # Find all links to individual sound pages
        sound_page_links = await page.eval_on_selector_all('a[href$=".html"]', 'elements => elements.map(el => el.href)')
        print(f"Found {len(sound_page_links)} sound page links.")

        # Create a directory to store the downloaded files
        if not os.path.exists("wav_files"):
            os.makedirs("wav_files")

        async with httpx.AsyncClient() as client:
            for sound_page_link in sound_page_links:
                if "tags-" in sound_page_link:
                    continue
                try:
                    print(f"Processing page: {sound_page_link}")
                    await page.goto(sound_page_link)
                    
                    # Corrected selector
                    wav_links = await page.eval_on_selector_all('a[href*="grab.php"][href*="type=wav"]', 'elements => elements.map(el => el.href)')
                    
                    print(f"Found {len(wav_links)} .wav links on this page.")

                    for link in wav_links:
                        try:
                            print(f"Downloading {link}")
                            response = await client.get(link, follow_redirects=True)
                            response.raise_for_status()
                            
                            title = await page.title()
                            sanitized_title = sanitize_filename(title)
                            file_name = os.path.join("wav_files", f"{sanitized_title}.wav")

                            with open(file_name, "wb") as f:
                                f.write(response.content)
                            print(f"Downloaded {file_name}")
                            
                            # Add a small delay
                            time.sleep(1)

                        except httpx.HTTPStatusError as e:
                            print(f"Failed to download {link}: HTTP Status {e.response.status_code}")
                        except Exception as e:
                            print(f"Failed to download {link}: {e}")
                except Exception as e:
                    print(f"Failed to process page {sound_page_link}: {e}")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
