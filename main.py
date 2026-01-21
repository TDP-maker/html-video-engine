import os
import asyncio
import sys
from playwright.async_api import async_playwright

# --- CONFIG ---
FPS = 24
DURATION = 3 
TOTAL_FRAMES = FPS * DURATION

# 1. CREATE A DUMMY HTML FILE (To ensure we have something to record)
# This removes "missing template" errors from the equation.
def create_test_html():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body { margin: 0; background: #111; display: flex; justify-content: center; align-items: center; height: 100vh; color: white; font-family: sans-serif; }
            .box { width: 200px; height: 200px; background: #00ff9d; animation: spin 3s linear infinite; }
            @keyframes spin { 100% { transform: rotate(360deg); } }
        </style>
    </head>
    <body>
        <div class="box"></div>
        <h1 style="position:absolute; bottom: 50px;">TEST RENDER</h1>
        <script>
            // The engine looks for this function to control time
            window.seekToFrame = (frame, fps) => {
                const time = frame / fps;
                document.getAnimations().forEach(anim => {
                    anim.currentTime = time * 1000;
                    anim.pause();
                });
            }
        </script>
    </body>
    </html>
    """
    with open("test_engine.html", "w") as f:
        f.write(html_content)
    print("✅ Created test_engine.html")

async def run_diagnostic():
    print("🛑 RESETTING ENVIRONMENT...")
    
    # Create test HTML
    create_test_html()
    
    current_folder = os.getcwd()
    output_folder = os.path.join(current_folder, "frames")
    
    # Clean render folder
    if not os.path.exists(output_folder): os.makedirs(output_folder)

    print("🚀 Launching Recorder...")
    async with async_playwright() as p:
        try:
            # We use the system executable because we know it exists in the nixpacks image
            browser = await p.chromium.launch(
                executable_path="/usr/bin/chromium",
                args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage", "--disable-gpu"]
            )
            # Mobile Portrait Resolution
            page = await browser.new_page(viewport={"width": 720, "height": 1280})
            
            # Load the dummy file we just made
            await page.goto(f"file://{os.path.join(current_folder, 'test_engine.html')}")
            
            print("📸 Recording Frames...")
            for frame in range(TOTAL_FRAMES):
                await page.evaluate(f"if(window.seekToFrame) window.seekToFrame({frame}, {FPS})")
                await page.screenshot(path=f"{output_folder}/frame_{frame:04d}.png")
            
            await browser.close()
            print("✅ Frames captured successfully.")
            
        except Exception as e:
            print(f"❌ CRITICAL ERROR IN BROWSER: {e}")
            # We don't exit; we keep the server running so you can see the logs
            pass

    # Stitch Video
    print("🔨 Stitching Video...")
    video_name = "proof_of_concept.mp4"
    exit_code = os.system(f"ffmpeg -y -r {FPS} -i {output_folder}/frame_%04d.png -vcodec libx264 -pix_fmt yuv420p {video_name}")
    
    if exit_code == 0:
        print("✅ VIDEO CREATED.")
    else:
        print("❌ FFMPEG FAILED.")

    # START SERVER
    print("\n" + "="*40)
    print(f"🌍 SYSTEM READY. CLICK YOUR DOMAIN LINK.")
    print(f"👇 You should see '{video_name}' in the list.")
    print("="*40 + "\n")
    
    port = int(os.environ.get("PORT", 8080))
    os.system(f"python -m http.server {port}")

if __name__ == "__main__":
    asyncio.run(run_diagnostic())
