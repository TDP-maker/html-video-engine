import os
import asyncio
import time
from playwright.async_api import async_playwright
from pyairtable import Table

AIRTABLE_API_KEY = os.environ.get("AIRTABLE_API_KEY")
BASE_ID = os.environ.get("AIRTABLE_BASE_ID")
TABLE_NAME = "Video Jobs"

FPS = 30
DURATION = 5
TOTAL_FRAMES = FPS * DURATION

async def render_job(job):
    print(f"🎬 Starting render for: {job['fields'].get('Product Name')}")
    current_folder = os.getcwd()
    template_url = "file://" + os.path.join(current_folder, "templates", "engine.html")
    output_folder = os.path.join(current_folder, "frames")
    if not os.path.exists(output_folder): os.makedirs(output_folder)

    title = job['fields'].get('Product Name', 'Unknown')
    price = job['fields'].get('Price', '')
    img_url = job['fields'].get('Image URL', '')

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1080, "height": 1920})
        full_url = f"{template_url}?title={title}&price={price}&img={img_url}"
        await page.goto(full_url)

        for frame in range(TOTAL_FRAMES):
            await page.evaluate(f"window.seekToFrame({frame}, {FPS})")
            await page.screenshot(path=f"{output_folder}/frame_{frame:04d}.png", type='png')
        await browser.close()

    video_name = f"video_{job['id']}.mp4"
    os.system(f"ffmpeg -y -r {FPS} -i {output_folder}/frame_%04d.png -vcodec libx264 -pix_fmt yuv420p {video_name}")
    print(f"✅ Video Created: {video_name}")
    return video_name

def run_worker():
    print("👷 Video Worker Started. Waiting for jobs...")
    if not AIRTABLE_API_KEY:
        print("❌ ERROR: AIRTABLE_API_KEY is missing!")
        return
    table = Table(AIRTABLE_API_KEY, BASE_ID, TABLE_NAME)

    while True:
        try:
            pending_jobs = table.all(formula="{Status}='Pending'")
            if pending_jobs:
                for job in pending_jobs:
                    asyncio.run(render_job(job))
                    table.update(job['id'], {"Status": "Done"})
            else:
                print("...no jobs found, sleeping...")
        except Exception as e:
            print(f"⚠️ Error: {e}")
        time.sleep(60)

if __name__ == "__main__":
    run_worker()
