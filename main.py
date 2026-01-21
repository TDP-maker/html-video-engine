import os
import asyncio
import subprocess
import imageio_ffmpeg
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from playwright.async_api import async_playwright
import uvicorn

app = FastAPI()

# --- THE DATA MODEL ---
class VideoBrief(BaseModel):
    title: str = "Default Title"
    cta: str = "Click Here"
    img: str = "https://images.unsplash.com/photo-1542291026-7eec264c27ff"
    color: str = "#00ff9d"
    format: str = "reel" # or "square"

# --- THE RENDER ENGINE ---
async def render_video(brief: VideoBrief):
    width, height = (1080, 1080) if brief.format == "square" else (1080, 1920)
    fps = 30
    duration = 5
    
    current_folder = os.getcwd()
    output_folder = os.path.join(current_folder, "frames")
    if not os.path.exists(output_folder): os.makedirs(output_folder)

    # Encode data for the HTML template
    query = f"?title={brief.title.replace(' ', '+')}&cta={brief.cta.replace(' ', '+')}&img={brief.img}&color={brief.color.replace('#', '%23')}"
    template_url = f"file://{os.path.join(current_folder, 'templates', 'engine.html')}{query}"

    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--no-sandbox"])
        page = await browser.new_page(viewport={"width": width, "height": height})
        await page.goto(template_url)
        
        for frame in range(fps * duration):
            await page.evaluate(f"window.seekToFrame({frame}, {fps})")
            await page.screenshot(path=f"{output_folder}/frame_{frame:04d}.png")
        await browser.close()

    video_name = f"final_output.mp4"
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg_exe, "-y", "-r", str(fps), "-i", f"{output_folder}/frame_%04d.png", "-vcodec", "libx264", "-pix_fmt", "yuv420p", video_name])
    print(f"✅ Render Complete: {video_name}")

# --- THE ENDPOINTS ---

@app.get("/")
async def home():
    return {"status": "Online", "message": "Video Engine Ready for Make.com"}

@app.post("/generate")
async def generate(brief: VideoBrief, background_tasks: BackgroundTasks):
    # This runs the render in the background so Make.com doesn't time out
    background_tasks.add_task(render_video, brief)
    return {"message": "Render started", "status": "processing"}

@app.get("/download")
async def download():
    return FileResponse("final_output.mp4", media_type="video/mp4", filename="creative.mp4")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
