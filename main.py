import os
import asyncio
import subprocess
import json
from playwright.async_api import async_playwright

# --- CONFIG ---
FPS = 15              # Standard Speed
DURATION = 3          # 3 Seconds
TOTAL_FRAMES = FPS * DURATION

async def run_test():
    print("🧪 STARTING RENDER + UPLOAD TEST...")
    
    current_folder = os.getcwd()
    template_url = "file://" + os.path.join(current_folder, "templates", "engine.html")
    output_folder = os.path.join(current_folder, "frames")
    if not os.path.exists(output_folder): os.makedirs(output_folder)

    # 1. GENERATE VIDEO
    print("🚀 Launching Chrome...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--no-sandbox", "--disable-setuid-sandbox"])
        page = await browser.new_page(viewport={"width": 720, "height": 1280})
        
        await page.goto(template_url)
        
        print("📸 Recording Frames...")
        for frame in range(TOTAL_FRAMES):
            await page.evaluate(f"if(window.seekToFrame) window.seekToFrame({frame}, {FPS})")
            await page.screenshot(path=f"{output_folder}/frame_{frame:04d}.png")
        
        await browser.close()

    # 2. STITCH VIDEO
    print("🔨 Stitching video...")
    video_name = "test_output.mp4"
    os.system(f"ffmpeg -y -r {FPS} -i {output_folder}/frame_%04d.png -vcodec libx264 -pix_fmt yuv420p {video_name}")
    
    # 3. UPLOAD VIDEO (The New Part)
    print("📤 Uploading video to File.io so you can see it...")
    
    # We use 'curl' to send the file to a free hosting service
    result = subprocess.run(
        ["curl", "-F", f"file=@{video_name}", "https://file.io"],
        capture_output=True, text=True
    )
    
    try:
        response = json.loads(result.stdout)
        if response.get("success"):
            print("\n" + "="*40)
            print(f"🎉 VIDEO READY! CLICK HERE TO WATCH:")
            print(f"👉 {response.get('link')}")
            print("="*40 + "\n")
        else:
            print("❌ Upload failed:", result.stdout)
    except:
        print("⚠️ Raw Upload Output:", result.stdout)

if __name__ == "__main__":
    asyncio.run(run_test())
