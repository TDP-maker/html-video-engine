import os
import asyncio
from playwright.async_api import async_playwright

# --- CONFIG ---
FPS = 10              # Low FPS for safety
DURATION = 2          # Short duration
TOTAL_FRAMES = FPS * DURATION

async def run_test():
    print("🧪 STARTING ANTI-CRASH TEST...")
    
    current_folder = os.getcwd()
    # Debug: Print where we are looking
    print(f"📂 Working Directory: {current_folder}")
    
    template_path = os.path.join(current_folder, "templates", "engine.html")
    template_url = "file://" + template_path
    output_folder = os.path.join(current_folder, "frames")
    
    if not os.path.exists(output_folder): os.makedirs(output_folder)

    print("🚀 Launching Chrome with Anti-Crash Flags...")
    async with async_playwright() as p:
        try:
            # THIS IS THE FIX: The args list tells Chrome it's okay to run on a server
            browser = await p.chromium.launch(
                args=["--no-sandbox", "--disable-setuid-sandbox"]
            )
            
            page = await browser.new_page(viewport={"width": 720, "height": 1280})
            
            print(f"🔗 Loading: {template_url}")
            await page.goto(template_url)
            
            # Record frames
            print("📸 Taking pictures...")
            for frame in range(TOTAL_FRAMES):
                await page.evaluate(f"if(window.seekToFrame) window.seekToFrame({frame}, {FPS})")
                await page.screenshot(path=f"{output_folder}/frame_{frame:04d}.png")
                
            print("✅ Photos captured successfully.")
            await browser.close()
            
        except Exception as e:
            print(f"❌ BROWSER CRASHED: {e}")
            # Print more details to help debug
            import traceback
            traceback.print_exc()
            return

    # Stitch Video
    print("🔨 Stitching video...")
    os.system(f"ffmpeg -y -r {FPS} -i {output_folder}/frame_%04d.png -vcodec libx264 -pix_fmt yuv420p output.mp4")
    print("🎉 SUCCESS! The engine is working.")

if __name__ == "__main__":
    asyncio.run(run_test())
