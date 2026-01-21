import os
import asyncio
import sys
import subprocess
from playwright.async_api import async_playwright
import imageio_ffmpeg

# --- CONFIG ---
FPS = 24
DURATION = 3 
TOTAL_FRAMES = FPS * DURATION

# 1. CREATE DUMMY HTML (So we don't need external templates)
def create_test_html():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body { margin: 0; background: #000; display: flex; justify-content: center; align-items: center; height: 100vh; color: #00ff9d; font-family: monospace; font-size: 40px; }
            .box { border: 5px solid #00ff9d; padding: 20px; animation: pulse 1s infinite; }
            @keyframes pulse { 50% { opacity: 0.5; } }
        </style>
    </head>
    <body>
        <div class="box">SYSTEM ONLINE</div>
        <script>
            window.seekToFrame = (frame, fps) => { 
                // Simple animation control
                document.body.style.opacity = (frame % fps) / fps;
            }
        </script>
    </body>
    </html>
    """
    with open("test_engine.html", "w") as f:
        f.write(html_content)

async def run_self_reliant_test():
    print("🛡️ STARTING SELF-RELIANT DIAGNOSTIC...")

    # 1. SETUP
    create_test_html()
    current_folder = os.getcwd()
    output_folder = os.path.join(current_folder, "frames")
    if not os.path.exists(output_folder): os.makedirs(output_folder)

    # 2. FIND FFMPEG (The Fix for 'ffmpeg not found')
    # This asks the library where its portable ffmpeg is hiding
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"✅ FOUND PORTABLE FFMPEG AT: {ffmpeg_exe}")

    # 3. LAUNCH CHROME
    print("🚀 Launching Browser...")
    async with async_playwright() as p:
        try:
            # We do NOT use 'executable_path' here. 
            # We let Playwright use the one we are about to download in the start command.
            browser = await p.chromium.launch(
                args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage", "--disable-gpu"]
            )
            page = await browser.new_page(viewport={"width": 720, "height": 1280})
            await page.goto(f"file://{os.path.join(current_folder, 'test_engine.html')}")
            
            print("📸 Recording...")
            for frame in range(TOTAL_FRAMES):
                await page.evaluate(f"if(window.seekToFrame) window.seekToFrame({frame}, {FPS})")
                await page.screenshot(path=f"{output_folder}/frame_{frame:04d}.png")
            
            await browser.close()
            print("✅ Frames captured.")
            
        except Exception as e:
            print(f"❌ BROWSER FAIL: {e}")
            return

    # 4. STITCH VIDEO
    print("🔨 Stitching Video...")
    video_name = "self_reliant.mp4"
    
    # We run the portable ffmpeg manually
    cmd = [
        ffmpeg_exe, "-y", 
        "-r", str(FPS), 
        "-i", f"{output_folder}/frame_%04d.png", 
        "-vcodec", "libx264", 
        "-pix_fmt", "yuv420p", 
        video_name
    ]
    subprocess.run(cmd)
    
    # 5. HOST IT
    print("\n" + "="*40)
    print(f"🌍 SYSTEM READY. CLICK YOUR DOMAIN LINK.")
    print(f"👇 You should see '{video_name}' in the list.")
    print("="*40 + "\n")
    
    port = int(os.environ.get("PORT", 8080))
    os.system(f"python -m http.server {port}")

if __name__ == "__main__":
    asyncio.run(run_self_reliant_test())
