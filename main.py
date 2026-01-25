import os
import uuid
import asyncio
import subprocess
import imageio_ffmpeg
import httpx
from fastapi import FastAPI, BackgroundTasks, HTTPException, Header, Depends
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from playwright.async_api import async_playwright
import uvicorn
from openai import OpenAI
import re
from datetime import datetime

app = FastAPI()

# Store video status
video_jobs = {}

# --- API KEY AUTHENTICATION ---
API_KEY = os.environ.get("API_KEY", "")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    """Verify API key for protected endpoints"""
    if not API_KEY:
        # No API key set = no protection (for development)
        return True
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True

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
    record_id: str = ""  # Airtable record ID for tracking

# --- RENDER ENGINES ---

async def render_video(brief: VideoBrief, video_id: str = None):
    """Original template-based renderer"""
    if not video_id:
        video_id = str(uuid.uuid4())[:8]

    width, height = (1080, 1080) if brief.format == "square" else (1080, 1920)
    fps = 30
    duration = 5

    current_folder = os.getcwd()
    output_folder = os.path.join(current_folder, "frames", video_id)
    videos_folder = os.path.join(current_folder, "videos")

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(videos_folder, exist_ok=True)

    video_jobs[video_id] = {"status": "rendering", "progress": 0}

    query = f"?title={brief.title.replace(' ', '+')}&cta={brief.cta.replace(' ', '+')}&img={brief.img}&color={brief.color.replace('#', '%23')}"
    template_url = f"file://{os.path.join(current_folder, 'templates', 'engine.html')}{query}"

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(args=["--no-sandbox"])
            page = await browser.new_page(viewport={"width": width, "height": height})
            await page.goto(template_url)

            total_frames = fps * duration
            for frame in range(total_frames):
                await page.evaluate(f"window.seekToFrame({frame}, {fps})")
                await page.screenshot(path=f"{output_folder}/frame_{frame:04d}.png")
                video_jobs[video_id]["progress"] = int((frame / total_frames) * 100)
            await browser.close()

        video_name = f"{videos_folder}/{video_id}.mp4"
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([ffmpeg_exe, "-y", "-r", str(fps), "-i", f"{output_folder}/frame_%04d.png",
                       "-vcodec", "libx264", "-pix_fmt", "yuv420p", video_name])

        video_jobs[video_id] = {"status": "complete", "progress": 100, "file": video_name}
        print(f"✅ Render Complete: {video_id}")

        # Cleanup frames
        for f in os.listdir(output_folder):
            os.remove(os.path.join(output_folder, f))
        os.rmdir(output_folder)

    except Exception as e:
        video_jobs[video_id] = {"status": "error", "error": str(e)}
        print(f"❌ Render Error: {e}")


async def render_video_from_html(html_content: str, video_id: str, format: str = "reel", fps: int = 30):
    """Render video from custom HTML with multiple frames"""
    width, height = (1080, 1080) if format == "square" else (1080, 1920)

    current_folder = os.getcwd()
    output_folder = os.path.join(current_folder, "frames", video_id)
    videos_folder = os.path.join(current_folder, "videos")

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(videos_folder, exist_ok=True)

    video_jobs[video_id] = {"status": "rendering", "progress": 0}

    # Save HTML to temp file
    html_path = os.path.join(current_folder, f"temp_{video_id}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(args=["--no-sandbox"])
            page = await browser.new_page(viewport={"width": width, "height": height})
            await page.goto(f"file://{html_path}")
            await page.wait_for_timeout(1000)

            has_frames = await page.evaluate("() => document.querySelectorAll('.frame').length > 0")

            frame_index = 0
            if has_frames:
                frame_count = await page.evaluate("() => document.querySelectorAll('.frame').length")
                print(f"📊 [{video_id}] Detected {frame_count} frames")

                timing = await page.evaluate("""() => {
                    return window.timing || Array(document.querySelectorAll('.frame').length).fill(3000);
                }""")

                for slide_num in range(frame_count):
                    await page.evaluate(f"""(slideNum) => {{
                        document.querySelectorAll('.frame').forEach((f, i) => {{
                            f.classList.remove('active', 'exit');
                            if (i === slideNum) f.classList.add('active');
                        }});
                    }}""", slide_num)

                    await page.wait_for_timeout(600)

                    slide_duration_ms = timing[slide_num] if slide_num < len(timing) else 3000
                    video_frames_for_slide = int((slide_duration_ms / 1000) * fps)

                    for i in range(video_frames_for_slide):
                        await page.screenshot(path=f"{output_folder}/frame_{frame_index:04d}.png")
                        frame_index += 1
                        await page.wait_for_timeout(int(1000 / fps))

                    video_jobs[video_id]["progress"] = int((slide_num / frame_count) * 100)
                    print(f"✅ [{video_id}] Slide {slide_num + 1}/{frame_count}")

            else:
                duration = 5
                total_frames = fps * duration
                print(f"📊 [{video_id}] Single frame mode")

                has_seek = await page.evaluate("() => typeof window.seekToFrame === 'function'")

                for frame in range(total_frames):
                    if has_seek:
                        await page.evaluate(f"window.seekToFrame({frame}, {fps})")
                    else:
                        await page.wait_for_timeout(int(1000 / fps))
                    await page.screenshot(path=f"{output_folder}/frame_{frame_index:04d}.png")
                    frame_index += 1
                    video_jobs[video_id]["progress"] = int((frame / total_frames) * 100)

            await browser.close()

        # Compile video
        video_name = f"{videos_folder}/{video_id}.mp4"
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([
            ffmpeg_exe, "-y", "-r", str(fps),
            "-i", f"{output_folder}/frame_%04d.png",
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "fast", "-crf", "23",
            video_name
        ])

        video_jobs[video_id] = {"status": "complete", "progress": 100, "file": video_name}
        print(f"✅ [{video_id}] Render Complete")

        # Cleanup
        for f in os.listdir(output_folder):
            os.remove(os.path.join(output_folder, f))
        os.rmdir(output_folder)
        if os.path.exists(html_path):
            os.remove(html_path)

    except Exception as e:
        video_jobs[video_id] = {"status": "error", "error": str(e)}
        print(f"❌ [{video_id}] Error: {e}")


async def extract_images_with_playwright(url: str) -> tuple[list, str, str]:
    """Use Playwright to load page and extract images from rendered DOM"""
    product_images = []
    page_title = ""
    page_description = ""

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(args=["--no-sandbox"])
            page = await browser.new_page()

            await page.goto(url, wait_until="networkidle", timeout=30000)
            await page.wait_for_timeout(2000)  # Extra wait for lazy-loaded images

            # Get page title and description
            page_title = await page.title()
            meta_desc = await page.query_selector('meta[name="description"]')
            if meta_desc:
                page_description = await meta_desc.get_attribute("content") or ""

            # Extract images from rendered DOM
            images = await page.evaluate("""() => {
                const images = [];

                // Get og:image first (usually best quality)
                const ogImage = document.querySelector('meta[property="og:image"]');
                if (ogImage) images.push(ogImage.content);

                // Get all img elements
                document.querySelectorAll('img').forEach(img => {
                    let src = img.src || img.dataset.src || img.getAttribute('data-lazy-src');
                    if (src && src.startsWith('http')) {
                        // Filter out tiny images, icons, tracking pixels
                        const width = img.naturalWidth || img.width || 0;
                        const height = img.naturalHeight || img.height || 0;
                        if (width >= 200 || height >= 200 || src.includes('product') || src.includes('hero')) {
                            images.push(src);
                        }
                    }
                });

                // Get background images from divs
                document.querySelectorAll('[style*="background"]').forEach(el => {
                    const style = el.getAttribute('style');
                    const match = style.match(/url\\(['"]?(https?[^'"\\)]+)['"]?\\)/);
                    if (match) images.push(match[1]);
                });

                return [...new Set(images)]; // Remove duplicates
            }""")

            await browser.close()

            # Filter images
            for img in images:
                if any(skip in img.lower() for skip in ['icon', 'logo', 'sprite', '.svg', 'pixel', 'tracking', '1x1', 'spacer', 'blank']):
                    continue
                if img not in product_images:
                    product_images.append(img)

            product_images = product_images[:10]  # Top 10 images

    except Exception as e:
        print(f"⚠️ Playwright extraction error: {e}")

    return product_images, page_title, page_description


async def generate_html_from_url(url: str, prompt: str = "") -> str:
    """Use Playwright + OpenAI to generate HTML video content from a URL"""

    print(f"🔍 Extracting images with Playwright...")
    product_images, page_title, page_description = await extract_images_with_playwright(url)
    print(f"📸 Found {len(product_images)} product images")

    # Also fetch raw HTML for text content
    base_url = '/'.join(url.split('/')[:3])
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, follow_redirects=True, timeout=30)
            page_content = response.text[:10000]
        except Exception as e:
            page_content = ""

    client = OpenAI()

    system_prompt = """You are a professional video ad designer creating high-end product videos for Instagram/TikTok.

Create a POLISHED, AD-READY HTML video with these specifications:

STRUCTURE:
- Use .frame class for each slide (5-7 frames)
- First frame: .frame.active (starts visible)
- Add: const timing = [3000, 3500, 3500, 3500, 3500, 3000];
- Container: .reel-container at 1080x1920px
- All frames: opacity:0 by default, .frame.active has opacity:1

PRODUCT IMAGES - MUST USE:
- Display the provided product images LARGE (500-700px width)
- Center images with clean spacing
- Add subtle drop-shadow: 0 20px 60px rgba(0,0,0,0.3)
- Use object-fit: contain to preserve aspect ratio
- Animate images: subtle float, scale, or fade effects

PROFESSIONAL STYLING:
- Font: 'Inter' or 'Plus Jakarta Sans' from Google Fonts
- Clean, minimal design - lots of whitespace
- Solid or subtle gradient backgrounds (not busy)
- Text should be large and readable (50-80px headlines)
- Use brand colors if detectable, otherwise elegant neutrals
- NO emojis unless specifically requested

ANIMATIONS:
- Smooth CSS transitions (0.5-1s duration)
- Subtle movements - don't overdo it
- Images: gentle scale (1 to 1.02) or float (translateY)
- Text: fade in, slide up from bottom
- Use @keyframes with ease-out timing

FRAME CONTENT:
1. HOOK: Large product image + short punchy headline (3-5 words)
2. FEATURE 1: Product image + key benefit
3. FEATURE 2: Another angle/product + second benefit
4. SOCIAL PROOF: Testimonial style or key stat
5. CTA: Product image + "Shop Now" / "Learn More" button style

Return ONLY valid HTML code, no explanations."""

    # Build image list
    images_text = "\n".join([f"{i+1}. {img}" for i, img in enumerate(product_images)]) if product_images else "No images found - use placeholder styling"

    user_prompt = f"""Create a premium product video ad:

BRAND: {page_title}
DESCRIPTION: {page_description}
URL: {url}

PRODUCT IMAGES TO USE (these are real, working URLs - use them!):
{images_text}

{f"SPECIAL INSTRUCTIONS: {prompt}" if prompt else ""}

Create a clean, professional, ad-ready HTML video showcasing these products. Make it look like a high-end brand advertisement."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=8000
    )

    html_content = response.choices[0].message.content
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
            "/generate": "POST - Template video",
            "/generate-html": "POST - Video from custom HTML",
            "/generate-from-url": "POST - AI generates video from URL",
            "/status/{video_id}": "GET - Check video status",
            "/download/{video_id}": "GET - Download specific video",
            "/download": "GET - Download latest video"
        }
    }

@app.post("/generate")
async def generate(brief: VideoBrief, background_tasks: BackgroundTasks, _: bool = Depends(verify_api_key)):
    """Original endpoint - template-based video (API key required)"""
    video_id = str(uuid.uuid4())[:8]
    background_tasks.add_task(render_video, brief, video_id)
    return {
        "message": "Render started",
        "status": "processing",
        "video_id": video_id,
        "format": brief.format
    }

@app.post("/generate-html")
async def generate_html_video(request: HTMLVideoRequest, background_tasks: BackgroundTasks, _: bool = Depends(verify_api_key)):
    """Render custom HTML as video (API key required)"""
    video_id = str(uuid.uuid4())[:8]
    background_tasks.add_task(render_video_from_html, request.html, video_id, request.format, request.fps)
    return {
        "message": "HTML render started",
        "status": "processing",
        "video_id": video_id,
        "format": request.format
    }

@app.post("/generate-from-url")
async def generate_from_url(request: URLVideoRequest, background_tasks: BackgroundTasks, _: bool = Depends(verify_api_key)):
    """AI generates HTML from URL, then renders as video (API key required)"""
    video_id = str(uuid.uuid4())[:8]

    try:
        print(f"🔍 [{video_id}] Analyzing URL: {request.url}")
        video_jobs[video_id] = {"status": "generating_html", "progress": 0}

        html_content = await generate_html_from_url(request.url, request.prompt)
        print(f"✅ [{video_id}] HTML generated ({len(html_content)} chars)")

        background_tasks.add_task(render_video_from_html, html_content, video_id, request.format, request.fps)

        return {
            "message": "Video generation started",
            "status": "processing",
            "video_id": video_id,
            "record_id": request.record_id,
            "url": request.url,
            "format": request.format
        }
    except Exception as e:
        video_jobs[video_id] = {"status": "error", "error": str(e)}
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{video_id}")
async def get_status(video_id: str):
    """Check video generation status"""
    if video_id not in video_jobs:
        raise HTTPException(status_code=404, detail="Video ID not found")

    job = video_jobs[video_id]
    response = {
        "video_id": video_id,
        "status": job["status"],
        "progress": job.get("progress", 0)
    }

    if job["status"] == "complete":
        response["download_url"] = f"/download/{video_id}"
    elif job["status"] == "error":
        response["error"] = job.get("error", "Unknown error")

    return response

@app.get("/download/{video_id}")
async def download_video(video_id: str):
    """Download a specific video by ID"""
    video_path = f"videos/{video_id}.mp4"
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(video_path, media_type="video/mp4", filename=f"{video_id}.mp4")

@app.get("/download")
async def download_latest():
    """Download the most recent video (backwards compatibility)"""
    videos_folder = "videos"
    if not os.path.exists(videos_folder):
        raise HTTPException(status_code=404, detail="No videos available")

    videos = [f for f in os.listdir(videos_folder) if f.endswith('.mp4')]
    if not videos:
        raise HTTPException(status_code=404, detail="No videos available")

    # Get most recent
    latest = max(videos, key=lambda x: os.path.getctime(os.path.join(videos_folder, x)))
    return FileResponse(f"{videos_folder}/{latest}", media_type="video/mp4", filename="creative.mp4")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
