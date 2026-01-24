import os
import asyncio
import subprocess
import imageio_ffmpeg
import httpx
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from playwright.async_api import async_playwright
import uvicorn
from openai import OpenAI
import re

app = FastAPI()

# --- DATA MODELS ---

class VideoBrief(BaseModel):
    """Original simple template-based video"""
    title: str = "Default Title"
    cta: str = "Click Here"
    img: str = "https://images.unsplash.com/photo-1542291026-7eec264c27ff"
    color: str = "#00ff9d"
    format: str = "reel"

class HTMLVideoRequest(BaseModel):
    """Generate video from HTML content"""
    html: str
    format: str = "reel"
    fps: int = 30

class URLVideoRequest(BaseModel):
    """Generate video from URL + prompt"""
    url: str
    prompt: str = ""
    format: str = "reel"
    fps: int = 30

# --- RENDER ENGINES ---

async def render_video(brief: VideoBrief):
    """Original template-based renderer"""
    width, height = (1080, 1080) if brief.format == "square" else (1080, 1920)
    fps = 30
    duration = 5

    current_folder = os.getcwd()
    output_folder = os.path.join(current_folder, "frames")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Clear old frames
    for f in os.listdir(output_folder):
        os.remove(os.path.join(output_folder, f))

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

    video_name = "final_output.mp4"
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg_exe, "-y", "-r", str(fps), "-i", f"{output_folder}/frame_%04d.png",
                   "-vcodec", "libx264", "-pix_fmt", "yuv420p", video_name])
    print(f"✅ Render Complete: {video_name}")


async def render_video_from_html(html_content: str, format: str = "reel", fps: int = 30):
    """Render video from custom HTML with multiple frames"""
    width, height = (1080, 1080) if format == "square" else (1080, 1920)

    current_folder = os.getcwd()
    output_folder = os.path.join(current_folder, "frames")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Clear old frames
    for f in os.listdir(output_folder):
        os.remove(os.path.join(output_folder, f))

    # Save HTML to temp file
    html_path = os.path.join(current_folder, "temp_render.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--no-sandbox"])
        page = await browser.new_page(viewport={"width": width, "height": height})
        await page.goto(f"file://{html_path}")

        # Wait for page to load
        await page.wait_for_timeout(1000)

        # Check if this HTML has a frame system (multiple slides)
        has_frames = await page.evaluate("() => document.querySelectorAll('.frame').length > 0")

        if has_frames:
            # Multi-frame HTML
            frame_count = await page.evaluate("() => document.querySelectorAll('.frame').length")
            print(f"📊 Detected {frame_count} frames in HTML")

            # Get timing array if exists
            timing = await page.evaluate("""() => {
                return window.timing || Array(document.querySelectorAll('.frame').length).fill(3000);
            }""")

            frame_index = 0
            for slide_num in range(frame_count):
                # Show this frame
                await page.evaluate(f"""(slideNum) => {{
                    document.querySelectorAll('.frame').forEach((f, i) => {{
                        f.classList.remove('active', 'exit');
                        if (i === slideNum) f.classList.add('active');
                    }});
                }}""", slide_num)

                # Wait for animation
                await page.wait_for_timeout(600)

                # Frames for this slide
                slide_duration_ms = timing[slide_num] if slide_num < len(timing) else 3000
                video_frames_for_slide = int((slide_duration_ms / 1000) * fps)

                for i in range(video_frames_for_slide):
                    await page.screenshot(path=f"{output_folder}/frame_{frame_index:04d}.png")
                    frame_index += 1
                    await page.wait_for_timeout(int(1000 / fps))

                print(f"✅ Captured slide {slide_num + 1}/{frame_count}")

        else:
            # Single frame or seekToFrame animation
            duration = 5
            total_frames = fps * duration
            print(f"📊 Single frame mode - {total_frames} frames")

            has_seek = await page.evaluate("() => typeof window.seekToFrame === 'function'")

            for frame in range(total_frames):
                if has_seek:
                    await page.evaluate(f"window.seekToFrame({frame}, {fps})")
                else:
                    await page.wait_for_timeout(int(1000 / fps))
                await page.screenshot(path=f"{output_folder}/frame_{frame:04d}.png")

        await browser.close()

    # Compile video
    video_name = "final_output.mp4"
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([
        ffmpeg_exe, "-y", "-r", str(fps),
        "-i", f"{output_folder}/frame_%04d.png",
        "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast", "-crf", "23",
        video_name
    ])
    print(f"✅ Render Complete: {video_name}")

    # Cleanup
    if os.path.exists(html_path):
        os.remove(html_path)


async def generate_html_from_url(url: str, prompt: str = "") -> str:
    """Use OpenAI to generate HTML video content from a URL"""

    # Fetch URL content
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, follow_redirects=True, timeout=30)
            page_content = response.text[:15000]
        except Exception as e:
            page_content = f"Could not fetch URL: {str(e)}"

    client = OpenAI()

    system_prompt = """You are an expert at creating animated HTML video content for Instagram/TikTok reels.

Create a multi-frame HTML video presentation with these requirements:

STRUCTURE:
- Use .frame class for each slide (5-7 frames)
- First frame should be .frame.active
- Include timing array: const timing = [3000, 4000, 4000, 4000, 4000, 3000];
- Container: .reel-container with 1080x1920px

STYLING:
- Modern Google Fonts (Plus Jakarta Sans, Inter, etc.)
- Brand colors extracted from the website
- Gradient backgrounds
- CSS animations (fadeIn, slideIn, scale, pulse)
- Emojis for visual interest
- Clean, bold typography

ANIMATIONS:
- Each .frame has opacity:0 by default, .frame.active has opacity:1
- Include @keyframes for transitions
- Smooth, professional animations

CONTENT FLOW:
1. Hook (attention-grabbing headline)
2. Problem/Pain points
3. Solution/Features
4. Social proof or benefits
5. Call-to-action

Return ONLY the complete HTML code."""

    user_prompt = f"""Create an animated HTML video for:

URL: {url}

Website content:
{page_content}

{f"Instructions: {prompt}" if prompt else ""}

Generate complete HTML with animations."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=8000
    )

    html_content = response.choices[0].message.content

    # Clean markdown code blocks
    html_content = re.sub(r'^```html?\n?', '', html_content)
    html_content = re.sub(r'\n?```$', '', html_content)

    return html_content


# --- API ENDPOINTS ---

@app.get("/")
async def home():
    return {
        "status": "Online",
        "message": "HTML Video Engine",
        "endpoints": {
            "/generate": "POST - Template video (title, cta, img, color)",
            "/generate-html": "POST - Video from custom HTML",
            "/generate-from-url": "POST - AI generates video from URL",
            "/download": "GET - Download video"
        }
    }

@app.post("/generate")
async def generate(brief: VideoBrief, background_tasks: BackgroundTasks):
    """Original endpoint - template-based video"""
    background_tasks.add_task(render_video, brief)
    return {"message": "Render started", "status": "processing", "format": brief.format}

@app.post("/generate-html")
async def generate_html_video(request: HTMLVideoRequest, background_tasks: BackgroundTasks):
    """Render custom HTML as video"""
    background_tasks.add_task(render_video_from_html, request.html, request.format, request.fps)
    return {"message": "HTML render started", "status": "processing", "format": request.format}

@app.post("/generate-from-url")
async def generate_from_url(request: URLVideoRequest, background_tasks: BackgroundTasks):
    """AI generates HTML from URL, then renders as video"""
    try:
        print(f"🔍 Analyzing URL: {request.url}")
        html_content = await generate_html_from_url(request.url, request.prompt)
        print(f"✅ HTML generated ({len(html_content)} chars)")

        background_tasks.add_task(render_video_from_html, html_content, request.format, request.fps)

        return {
            "message": "Video generation started",
            "status": "processing",
            "url": request.url,
            "format": request.format
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download")
async def download():
    video_path = "final_output.mp4"
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="No video available")
    return FileResponse(video_path, media_type="video/mp4", filename="creative.mp4")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
