import os
import asyncio
import subprocess
import json
import sys
from playwright.async_api import async_playwright

# --- SUPER LITE CONFIG ---
# We only render 5 frames total to stop the crash
TOTAL_FRAMES = 5  
FPS = 1

async def run_diagnostic():
    print("🚑 STARTING DIAGNOSTIC MODE...")
    
    # 1. PRINT SYSTEM INFO (Check if we have memory)
    try:
        print("📊 Checking System...")
        os.system("free -m") # Prints available RAM
    except:
        pass

    # 2. SETUP PATHS
    current_folder = os.getcwd()
    template_path = "file://" + os.path.join(current_folder, "templates", "engine.html")
    output_folder = os.path.join(current_folder, "frames")
    
    # Force clean start
    if not os.path.exists(output_folder): 
        os.makedirs(output_folder)
    
    # 3. LAUNCH CHROME (Minimal Mode)
    print("🚀 Attempting to launch Chrome...")
    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(
                args=[
                    "--no-sandbox", 
                    "--disable-setuid-sandbox", 
                    "--disable-dev-shm-usage",
                    "--disable-gpu"
                ]
            )
            # Tiny resolution to save RAM
            page = await browser.new_page(viewport={"width": 300, "height": 300})
            
            print(f"🔗 Loading HTML...")
            await page.goto(template_path)
            
            print("📸 Taking 5 Test Photos...")
            for frame in range(TOTAL_FRAMES):
                await page.evaluate(f"if(window.seekToFrame) window.seekToFrame({frame}, {FPS})")
                await page.screenshot(path=f"{output_folder}/frame_{frame:04d}.png")
                print(f"   - Frame {frame} saved")
            
            await browser.close()
            print("✅ Chrome worked!")
            
        except Exception as e:
            print(f"❌ CHROME DIED: {e}")
            sys.exit(1) # Stop here if Chrome fails

    # 4. STITCH VIDEO
    print("🔨 Making Video...")
    video_name = "mini_test.mp4"
    # Ultra-fast settings
    os.system(f"ffmpeg -y -r {FPS} -i {output_folder}/frame_%04d.png -vcodec libx264 -pix_fmt yuv420p -preset ultrafast {video_name}")
