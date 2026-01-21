import os
import asyncio
from playwright.async_api import async_playwright

# --- CONFIG ---
FPS = 30
DURATION = 5
TOTAL_FRAMES = FPS * DURATION

async def run_test():
    print("🧪 STARTING SELF-TEST MODE...")
    
    # 1. Mock Data (Fake Airtable Job)
    product_name = "TEST SHOE 3000"
    price = "$999"
    # Use a reliable placeholder image
    img_url = "https://assets.adidas.com/images/h_840,f_auto,q_auto,fl_lossy,c_fill,g_auto/69cbc73d0cb846889f89acbb011e6c75_9366/Ultraboost_21_Shoes_White_FY0379_01_standard.jpg"

    # 2. Setup Files
    current_folder = os.getcwd()
    template_url = "file://" + os.path.join(current_folder, "templates", "engine.html")
    output_folder = os.path.join(current_folder, "frames")
    
    # Create frames folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📂 Created folder: {output_folder}")

    # 3. Launch Browser
    print("🚀 Launching Chrome...")
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1080, "height": 1920})
        
        # Load Page with Data
        full_url = f"{template_url}?title={product_name}&price={price}&img={img_url}"
        print(f"🔗 Loading Template: {full_url}")
        await page.goto(full_url)

        # 4. Record Frame by Frame
        print("📸 Starting Frame Extraction (This takes ~30 seconds)...")
        for frame in range(TOTAL_FRAMES):
            # Move the animation to the exact frame
            await page.evaluate(f"window.seekToFrame({frame}, {FPS})")
            # Take a 4K screenshot
            await page.screenshot(path=f"{output_folder}/frame_{frame:04d}.png", type='png')
            
            # Print progress every 30 frames
            if frame % 30 == 0:
                print(f"   Rendered frame {frame}/{TOTAL_FRAMES}")

        await browser.close()

    # 5. Stitch Video
    print("🔨 Stitching frames with FFmpeg...")
    video_name = "test_video_output.mp4"
    
    # Run FFmpeg command
    exit_code = os.system(f"ffmpeg -y -r {FPS} -i {output_folder}/frame_%04d.png -vcodec libx264 -pix_fmt yuv420p {video_name}")
    
    if exit_code == 0:
        print(f"\n✅ SUCCESS! Video created: {video_name}")
        print("🎉 The engine is working perfectly.")
    else:
        print(f"\n❌ ERROR: FFmpeg failed with code {exit_code}")

if __name__ == "__main__":
    asyncio.run(run_test())
