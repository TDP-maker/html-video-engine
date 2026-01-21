import os
import threading
import http.server
import socketserver
import asyncio
import subprocess
import imageio_ffmpeg
from playwright.async_api import async_playwright

# --- CONFIG ---
FPS = 24
DURATION = 3
TOTAL_FRAMES = FPS * DURATION
PORT = int(os.environ.get("PORT", 8080))

# 1. START THE WEB SERVER IMMEDIATELY (To satisfy Railway)
def run_server():
    print(f"🌍 WEB SERVER STARTING ON PORT {PORT}...")
    # Create a simple "Loading" file so you see something
    with open("index.html", "w") as f:
        f.write("<h1>🚧 Installing Chrome & Rendering Video... Please Refresh in 30 seconds. 🚧</h1>")
        
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print("✅ SERVER IS LIVE. LISTENING FOR REQUESTS.")
        httpd.serve_forever()

# Start server in a background thread
server_thread = threading.Thread(target=run_server)
server_thread.daemon = True
server_thread.start()

# 2. THE WORKER FUNCTION (Downloads Chrome & Makes Video)
async def run_worker():
    print("⏳ STARTING BACKGROUND WORKER...")
    
    # A. INSTALL CHROME (Inside Python now!)
    print("⬇️ Downloading Chrome (This may take 60s)...")
    subprocess.run("playwright install chromium", shell=True)
    print("✅ Chrome Downloaded.")

    # B. CREATE DUMMY HTML
    with open("test_engine.html", "w") as f:
        f.write("<h1>SYSTEM ONLINE</h1><style>body{background:green;display:flex;justify-content:center;align-items:center;height:100vh;font-size:50px;font-family:sans-serif;}</style>")

    # C. RECORD
    current_folder = os.getcwd()
    output_folder = os.path.join(current_folder, "frames")
    if not os.path.exists(output_folder): os.makedirs(output_folder)
    
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    
    print("🚀 Launching Browser...")
    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage", "--disable-gpu"])
            page = await browser.new_page(viewport={"width": 720, "height": 1280})
            await page.goto(f"file://{os.path.join(current_folder, 'test_engine.html')}")
            
            print("📸 Recording frames...")
            for frame in range(TOTAL_FRAMES):
                await page.screenshot(path=f"{output_folder}/frame_{frame:04d}.png")
            await browser.close()
        except Exception as e:
            print(f"❌ ERROR: {e}")
            return

    # D. STITCH
    print("🔨 Stitching Video...")
    subprocess.run([ffmpeg_exe, "-y", "-r", str(FPS), "-i", f"{output_folder}/frame_%04d.png", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "final_video.mp4"])
    
    # E. UPDATE INDEX TO SHOW VIDEO
    with open("index.html", "w") as f:
        f.write("<h1>✅ DONE! <a href='final_video.mp4'>CLICK HERE TO WATCH</a></h1>")
    print("🎉 WORK COMPLETE.")

if __name__ == "__main__":
    # Run the worker, but the server is already running in the background thread!
    asyncio.run(run_worker())
    # Keep script alive
    while True: asyncio.sleep(1)
