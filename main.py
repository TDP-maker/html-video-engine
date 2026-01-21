import os
import threading
import http.server
import socketserver
import asyncio
import subprocess
import imageio_ffmpeg
from playwright.async_api import async_playwright

PORT = int(os.environ.get("PORT", 8080))

def run_server():
    with open("index.html", "w") as f:
        f.write("<h1>🎬 Rendering Video... Refresh in 30s.</h1>")
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        httpd.serve_forever()

threading.Thread(target=run_server, daemon=True).start()

async def run_worker():
    current_folder = os.getcwd()
    output_folder = os.path.join(current_folder, "frames")
    if not os.path.exists(output_folder): os.makedirs(output_folder)
    
    # Simple HTML for the video
    with open("test.html", "w") as f:
        f.write("<body style='background:black;color:white;display:flex;justify-content:center;align-items:center;height:100vh;'><h1>PROD READY</h1></body>")

    async with async_playwright() as p:
        try:
            # Docker image already has the browser in the right spot
            browser = await p.chromium.launch(args=["--no-sandbox"])
            page = await browser.new_page(viewport={"width": 720, "height": 1280})
            await page.goto(f"file://{os.path.join(current_folder, 'test.html')}")
            
            for frame in range(30):
                await page.screenshot(path=f"{output_folder}/frame_{frame:04d}.png")
            await browser.close()
            
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            subprocess.run([ffmpeg_exe, "-y", "-r", "15", "-i", f"{output_folder}/frame_%04d.png", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "final_video.mp4"])
            
            with open("index.html", "w") as f:
                f.write("<h1>✅ SUCCESS! <a href='final_video.mp4'>WATCH VIDEO</a></h1>")
        except Exception as e:
            with open("index.html", "w") as f:
                f.write(f"<h1>❌ ERROR: {str(e)}</h1>")

if __name__ == "__main__":
    asyncio.run(run_worker())
    while True: import time; time.sleep(1)
