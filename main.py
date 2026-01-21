import os
import asyncio
import subprocess
import sys
from playwright.async_api import async_playwright

# --- CONFIG ---
FPS = 15
DURATION = 3
TOTAL_FRAMES = FPS * DURATION

async def run_system_chrome_test():
    print("🤖 STARTING SYSTEM CHROME TEST...")
    
    current_folder = os.getcwd()
    template_path = "file://" + os.path.join(current_folder, "templates", "engine.html")
    output_folder = os.path.join(current_folder, "frames")
    if not os.path.exists(output_folder): os.makedirs(output_folder)

    # 1. LAUNCH PRE-INSTALLED CHROME
    print("🚀 Connecting to System Chrome...")
    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(
                # THIS IS THE FIX: Point to the installed Chrome
                executable_path="/usr/bin/chromium", 
                args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage", "--disable-gpu"]
            )
            page = await browser.new_page(viewport={"width": 720, "height": 1280})
            
            await page.goto(f"{template_path}?title=SYSTEM%20TEST&price=FREE")
            
            print(f"📸 Recording {TOTAL_FRAMES} frames...")
            for frame in range(TOTAL_FRAMES):
                await page.evaluate(f"if(window.seekToFrame) window.seekToFrame({frame}, {FPS})")
                await page.screenshot(path=f"{output_folder}/frame_{frame:04d}.png")
                if frame % 10 == 0: print(f"   Saved frame {frame}/{TOTAL_FRAMES}")
            
            await browser.close()
            
        except Exception as e:
            print(f"❌ CRASH: {e}")
            sys.exit(1)

    # 2. STITCH
    print("🔨 Stitching...")
    video_name = "system_test.mp4"
    os.system(f"ffmpeg -y -r {FPS} -i {output_folder}/frame_%04d.png -vcodec libx264 -pix_fmt yuv420p -preset ultrafast {video_name}")
    
    # 3. UPLOAD
    print("📤 Uploading...")
    try:
        cmd = f'curl --upload-file {video_name} https://transfer.sh/{video_name}'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        link = result.stdout.strip()
        print("\n" + "✅"*20)
        print(f"IT WORKS! CLICK HERE:")
        print(f"👉 {link}")
        print("✅"*20 + "\n")
    except Exception as e:
        print(f"❌ Upload failed: {e}")

if __name__ == "__main__":
    asyncio.run(run_system_chrome_test())
