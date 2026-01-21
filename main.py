import os
import threading
import http.server
import socketserver
import asyncio
import subprocess
import imageio_ffmpeg
from playwright.async_api import async_playwright

# --- CONFIGURATION ---
PORT = int(os.environ.get("PORT", 8080))
FPS = 30  # High quality 30fps
DURATION = 5 # 5 second video

async def run_production_render(job_data):
    # 1. SETUP DIMENSIONS
    # Toggle between Reel (9:16) and Square (1:1)
    if job_data.get("format") == "square":
        width, height = 1080, 1080
    else:
        width, height = 1080, 1920

    current_folder = os.getcwd()
    output_folder = os.path.join(current_folder, "frames")
    if not os.path.exists(output_folder): os.makedirs(output_folder)

    # 2. BUILD THE DATA URL
    # This passes your AI's brief into the HTML template
    query_params = f"?title={job_data['title']}&cta={job_data['cta']}&img={job_data['img']}&color={job_data['color']}"
    template_url = f"file://{os.path.join(current_folder, 'templates', 'engine.html')}{query_params}"

    async with async_playwright() as p:
        try:
            # Launching the Docker-provided Chromium
            browser = await p.chromium.launch(args=["--no-sandbox"])
            page = await browser.new_page(viewport={"width": width, "height": height})
            await page.goto(template_url)
            
            # 3. RECORDING
            total_frames = FPS * DURATION
            for frame in range(total_frames):
                # Tell the HTML to move to the specific frame
                await page.evaluate(f"if(window.seekToFrame) window.seekToFrame({frame}, {FPS})")
                await page.screenshot(path=f"{output_folder}/frame_{frame:04d}.png")
            
            await browser.close()

            # 4. STITCHING
            video_name = f"creative_{job_data['format']}.mp4"
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            subprocess.run([
                ffmpeg_exe, "-y", "-r", str(FPS), 
                "-i", f"{output_folder}/frame_%04d.png", 
                "-vcodec", "libx264", "-crf", "18", # High quality 18
                "-pix_fmt", "yuv420p", video_name
            ])
            
            # Update the web view
            with open("index.html", "w") as f:
                f.write(f"<h1>✅ CREATIVE READY!</h1><a href='{video_name}' style='font-size:40px'>Download {job_data['format'].upper()} Video</a>")
                
        except Exception as e:
            with open("index.html", "w") as f:
                f.write(f"<h1>❌ RENDER ERROR: {str(e)}</h1>")

# --- WEB INTERFACE ---
def run_server():
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        httpd.serve_forever()

if __name__ == "__main__":
    # Start the web server so Railway is happy
    threading.Thread(target=run_server, daemon=True).start()

    # MOCK DATA: In a real app, this comes from your Frontend/AI
    mock_brief = {
        "title": "FLASH SALE TODAY",
        "cta": "GET 50% OFF",
        "img": "https://images.unsplash.com/photo-1542291026-7eec264c27ff",
        "color": "%23ff0055", # Hot Pink
        "format": "reel" # Change to 'square' to test 1:1
    }

    # Start the render
    asyncio.run(run_production_render(mock_brief))
    
    # Keep alive
    while True: import time; time.sleep(1)
