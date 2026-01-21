import os
import threading
import http.server
import socketserver
import asyncio
import subprocess
import imageio_ffmpeg
import datetime
from playwright.async_api import async_playwright

# --- CONFIG ---
FPS = 24
DURATION = 3
TOTAL_FRAMES = FPS * DURATION
PORT = int(os.environ.get("PORT", 8080))

# HELPER: Write status to the website so you can see it
def update_status(message):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message) # Print to logs
    
    # Update index.html so the browser shows the latest step
    html = f"""
    <html>
    <head><meta http-equiv="refresh" content="2"></head> <body style="background: #111; color: #0f0; font-family: monospace; padding: 20px;">
        <h1>🚀 SYSTEM STATUS LOG</h1>
        <pre style="font-size: 18px; border: 1px solid #333; padding: 20px;">{full_message}</pre>
        <p>The page will auto-refresh...</p>
    </body>
    </html>
    """
    with open("index.html", "w") as f:
        f.write(html)

# 1. SERVER THREAD
def run_server():
    update_status("Initializing Web Server...")
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        httpd.serve_forever()

threading.Thread(target=run_server, daemon=True).start()

# 2. WORKER THREAD
async def run_worker():
    try:
        # A. INSTALL
        update_status("Step 1/5: Checking for Chrome...")
        # We try to install, capturing output to ensure it works
        proc = subprocess.run("playwright install chromium", shell=True, capture_output=True, text=True)
        if proc.returncode != 0:
            update_status(f"❌ INSTALL FAILED: {proc.stderr}")
            return
        update_status("✅ Chrome Installed. Ready to Launch.")

        # B. DUMMY FILE
        update_status("Step 2/5: Creating Test HTML...")
        with open("test_engine.html", "w") as f:
            f.write("<html><body style='background:red'><h1>IT WORKS</h1></body></html>")

        # C. RECORD
        current_folder = os.getcwd()
        output_folder = os.path.join(current_folder, "frames")
        if not os.path.exists(output_folder): os.makedirs(output_folder)
        
        update_status("Step 3/5: Launching Browser Engine...")
        async with async_playwright() as p:
            browser = await p.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"])
            page = await browser.new_page()
            await page.goto(f"file://{os.path.join(current_folder, 'test_engine.html')}")
            
            update_status(f"Step 4/5: Recording {TOTAL_FRAMES} frames...")
            for frame in range(TOTAL_FRAMES):
                await page.screenshot(path=f"{output_folder}/frame_{frame:04d}.png")
                # Update status every 10 frames so you know it's moving
                if frame % 10 == 0:
                    update_status(f"📸 Snapping frame {frame}/{TOTAL_FRAMES}...")
            
            await browser.close()

        # D. STITCH
        update_status("Step 5/5: Stitching Video with FFmpeg...")
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([ffmpeg_exe, "-y", "-r", str(FPS), "-i", f"{output_folder}/frame_%04d.png", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "final_video.mp4"])
        
        # E. SUCCESS
        final_html = """
        <html>
        <body style="background: #111; color: white; text-align: center; padding-top: 50px;">
            <h1 style="color: #0f0;">✅ SUCCESS!</h1>
            <a href="final_video.mp4" style="font-size: 30px; color: #00ff9d;">DOWNLOAD VIDEO</a>
        </body>
        </html>
        """
        with open("index.html", "w") as f:
            f.write(final_html)
            
    except Exception as e:
        update_status(f"❌ FATAL ERROR: {str(e)}")

if __name__ == "__main__":
    asyncio.run(run_worker())
    while True: asyncio.sleep(1)
