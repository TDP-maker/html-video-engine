import os
import asyncio
import sys
from playwright.async_api import async_playwright

# --- CONFIG (The one that worked) ---
TOTAL_FRAMES = 5  
FPS = 1

async def run_server_mode():
    print("🚑 REVERTING TO WORKING CODE...")
    
    # 1. SETUP
    current_folder = os.getcwd()
    template_path = "file://" + os.path.join(current_folder, "templates", "engine.html")
    output_folder = os.path.join(current_folder, "frames")
    if not os.path.exists(output_folder): os.makedirs(output_folder)
    
    # 2. GENERATE (Lite Mode)
    print("🚀 Making Video (Lite Mode)...")
    async with async_playwright() as p:
        try:
            # Using system chrome which we know works
            executable_path = "/usr/bin/chromium" 
            
            browser = await p.chromium.launch(
                executable_path=executable_path,
                args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage", "--disable-gpu"]
            )
            page = await browser.new_page(viewport={"width": 300, "height": 300})
            
            await page.goto(template_path)
            
            for frame in range(TOTAL_FRAMES):
                await page.evaluate(f"if(window.seekToFrame) window.seekToFrame({frame}, {FPS})")
                await page.screenshot(path=f"{output_folder}/frame_{frame:04d}.png")
            
            await browser.close()
            print("✅ Frames captured.")
            
        except Exception as e:
            print(f"❌ GENERATION ERROR: {e}")
            # If it fails, we still start the server so you can see logs
            pass

    # 3. STITCH
    video_name = "final_video.mp4"
    os.system(f"ffmpeg -y -r {FPS} -i {output_folder}/frame_%04d.png -vcodec libx264 -pix_fmt yuv420p {video_name}")
    print("✅ Video stitched.")

    # 4. START WEB SERVER (The New Way to View)
    print("\n" + "="*40)
    print(f"🌍 SERVER IS LIVE! CLICK YOUR RAILWAY DOMAIN TO DOWNLOAD.")
    print(f"👇 The video file is named: {video_name}")
    print("="*40 + "\n")
    
    # This command turns the folder into a website
    port = int(os.environ.get("PORT", 8080))
    os.system(f"python -m http.server {port}")

if __name__ == "__main__":
    asyncio.run(run_server_mode())
# ... previous code ...

# 4. START WEB SERVER
print(f"🌍 SERVER IS LIVE! CLICK YOUR RAILWAY DOMAIN TO DOWNLOAD.")
port = int(os.environ.get("PORT", 8080))
os.system(f"python -m http.server {port}")
