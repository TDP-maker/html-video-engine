import os
import asyncio
import subprocess
import sys
from playwright.async_api import async_playwright

# --- HD CONFIG ---
FPS = 24
DURATION = 4
TOTAL_FRAMES = FPS * DURATION

async def run_hd_test():
    print("🎬 STARTING HD RENDER (1080x1920)...")
    
    current_folder = os.getcwd()
    template_path = "file://" + os.path.join(current_folder, "templates", "engine.html")
    output_folder = os.path.join(current_folder, "frames")
    
    # Create folder if missing
    if not os.path.exists(output_folder): os.makedirs(output_folder)

    # 1. LAUNCH CHROME
    print("🚀 Launching Chrome...")
    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(
                args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage", "--disable-gpu"]
            )
            page = await browser.new_page(viewport={"width": 1080, "height": 1920})
            
            # Use fake data
            fake_url = f"{template_path}?title=CYBER%20BOOTS&price=$299"
            await page.goto(fake_url)
            
            print(f"📸 Recording {TOTAL_FRAMES} frames...")
            for frame in range(TOTAL_FRAMES):
                await page.evaluate(f"if(window.seekToFrame) window.seekToFrame({frame}, {FPS})")
                await page.screenshot(path=f"{output_folder}/frame_{frame:04d}.png")
                if frame % 10 == 0: print(f"   Saved frame {frame}/{TOTAL_FRAMES}")
            
            await browser.close()
        except Exception as e:
            print(f"❌ CRASH: {e}")
            sys.exit(1)

    # 2. STITCH VIDEO (Now safely inside the function!)
    print("🔨 Stitching HD Video...")
    video_name = "hd_test.mp4"
    
    # Run FFmpeg
    exit_code = os.system(f"ffmpeg -y -r {FPS} -i {output_folder}/frame_%04d.png -vcodec libx264 -pix_fmt yuv420p -preset veryfast {video_name}")
    
    if exit_code != 0:
        print("❌ FFmpeg failed to stitch video.")
        return

    # 3. UPLOAD TO TRANSFER.SH
    print("📤 Uploading...")
    try:
        cmd = f'curl --upload-file {video_name} https://transfer.sh/{video_name}'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        link = result.stdout.strip()
        print("\n" + "🎬"*20)
        print(f"YOUR HD VIDEO IS READY (Link valid for 14 days):")
        print(f"👉 {link}")
        print("🎬"*20 + "\n")
            
    except Exception as e:
        print(f"❌ Upload failed: {e}")

if __name__ == "__main__":
    asyncio.run(run_hd_test())
