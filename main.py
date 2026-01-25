import os
import uuid
import asyncio
import subprocess
import imageio_ffmpeg
import httpx
from fastapi import FastAPI, BackgroundTasks, HTTPException, Header, Depends
from fastapi.responses import FileResponse, HTMLResponse
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
                    slide_duration_ms = timing[slide_num] if slide_num < len(timing) else 3000

                    if slide_num == 0:
                        # First slide - just activate and capture
                        await page.evaluate("""() => {
                            document.querySelectorAll('.frame')[0].classList.add('active');
                        }""")
                        await page.wait_for_timeout(100)  # Let animations start

                        # Capture frames for this slide
                        slide_frames = int((slide_duration_ms / 1000) * fps)
                        for i in range(slide_frames):
                            await page.screenshot(path=f"{output_folder}/frame_{frame_index:04d}.png")
                            frame_index += 1
                            await page.wait_for_timeout(int(1000 / fps))
                    else:
                        # Transition: fade out previous, fade in current
                        await page.evaluate(f"""(slideNum) => {{
                            const frames = document.querySelectorAll('.frame');
                            // Start exit on previous
                            frames[slideNum - 1].classList.add('exit');
                            frames[slideNum - 1].classList.remove('active');
                            // Start enter on current
                            frames[slideNum].classList.add('active');
                        }}""", slide_num)

                        # Capture transition (1 second crossfade)
                        transition_duration = 1.0
                        transition_frames = int(transition_duration * fps)
                        for i in range(transition_frames):
                            await page.screenshot(path=f"{output_folder}/frame_{frame_index:04d}.png")
                            frame_index += 1
                            await page.wait_for_timeout(int(1000 / fps))

                        # Clean up exit class
                        await page.evaluate(f"""(slideNum) => {{
                            document.querySelectorAll('.frame')[slideNum - 1].classList.remove('exit');
                        }}""", slide_num)

                        # Capture hold frames (remaining duration minus transition)
                        hold_frames = int((slide_duration_ms / 1000) * fps) - transition_frames
                        for i in range(max(0, hold_frames)):
                            await page.screenshot(path=f"{output_folder}/frame_{frame_index:04d}.png")
                            frame_index += 1
                            await page.wait_for_timeout(int(1000 / fps))

                    video_jobs[video_id]["progress"] = int(((slide_num + 1) / frame_count) * 100)
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

    system_prompt = """You are a premium video ad designer. Create cinematic Instagram Reel HTML videos.

⚠️ INSTAGRAM SAFE ZONES - CRITICAL:
- TOP 250px: Username/follow button overlay - NO important content
- BOTTOM 400px: Captions/music/buttons overlay - NO important text here
- RIGHT 150px: Like/comment/share buttons - keep content left of this
- SAFE AREA: Content should be within x:0-930px, y:250-1520px

MANDATORY STRUCTURE (copy this exactly):
```
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap');
* { margin: 0; padding: 0; box-sizing: border-box; }

/* ANIMATED GRADIENT BACKGROUND */
body { background: #0a0a0a; }
.reel-container {
  width: 1080px; height: 1920px; position: relative; overflow: hidden;
  background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #0a0a0a 100%);
}
.bg-glow {
  position: absolute; width: 150%; height: 150%; top: -25%; left: -25%;
  background: radial-gradient(circle at 30% 20%, rgba(99,102,241,0.15) 0%, transparent 50%),
              radial-gradient(circle at 70% 80%, rgba(236,72,153,0.1) 0%, transparent 50%);
  animation: glowMove 8s ease-in-out infinite;
}
@keyframes glowMove {
  0%, 100% { transform: translate(0, 0) scale(1); }
  50% { transform: translate(30px, -30px) scale(1.1); }
}

/* FRAME TRANSITIONS - Respects Instagram safe zones */
.frame { position: absolute; top: 0; left: 0; width: 100%; height: 100%; opacity: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 250px 80px 400px 80px; transition: opacity 1s ease-in-out; }
.frame.active { opacity: 1; transition: opacity 1s ease-in-out; }
.frame.exit { opacity: 0; transition: opacity 1s ease-in-out; }

/* Safe zone helper - balanced vertical distribution */
.safe-zone { position: absolute; top: 200px; left: 80px; right: 180px; bottom: 350px; display: flex; flex-direction: column; align-items: center; justify-content: space-between; }
.content-area { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; width: 100%; }

/* PRODUCT ANIMATIONS */
.frame.active .product-wrap { animation: floatIn 1s cubic-bezier(0.34, 1.56, 0.64, 1) forwards, float 4s ease-in-out 1s infinite; }
.frame.active .text-area { animation: fadeUp 0.8s ease-out 0.4s forwards; opacity: 0; }
.frame.active .lifestyle-img { animation: zoomIn 1.2s ease-out forwards; }
.frame.active .accent-line { animation: lineGrow 0.6s ease-out 0.6s forwards; }

/* PRODUCT TREATMENT - Centered in safe zone */
.product-wrap { position: relative; transform: scale(0.85) translateY(40px); opacity: 0; z-index: 1; margin-bottom: 60px; }
.product-wrap::before {
  content: ''; position: absolute; top: 50%; left: 50%;
  transform: translate(-50%, -50%); width: 140%; height: 140%;
  background: radial-gradient(ellipse at center, rgba(255,255,255,0.95) 0%, rgba(240,240,240,0.7) 30%, rgba(150,150,150,0.2) 50%, transparent 70%);
  z-index: -1; border-radius: 50%;
}
.product-img { width: 800px; height: auto; max-height: 750px; object-fit: contain; filter: drop-shadow(0 50px 100px rgba(0,0,0,0.6)); }

/* LIFESTYLE TREATMENT */
.frame.lifestyle { padding-top: 0; padding-right: 0; }
.lifestyle-img { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; opacity: 0; transform: scale(1.08); }
.lifestyle-overlay { position: absolute; bottom: 0; left: 0; width: 100%; height: 70%; background: linear-gradient(to top, rgba(0,0,0,0.95) 0%, rgba(0,0,0,0.7) 40%, transparent 100%); z-index: 1; }

/* PREMIUM TEXT STYLING - 15% from bottom (288px), above Instagram UI */
.text-area { position: absolute; bottom: 300px; left: 0; text-align: center; width: 100%; padding: 0 120px; padding-right: 200px; transform: translateY(30px); z-index: 10; }
h1 {
  font-family: 'Inter', sans-serif; font-size: 72px; font-weight: 900;
  color: white; text-transform: uppercase; line-height: 1.1; letter-spacing: -1px;
  text-shadow: 0 4px 30px rgba(0,0,0,0.5);
}
.text-gradient {
  background: linear-gradient(135deg, #fff 0%, #a5b4fc 50%, #fff 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
}
p { font-family: 'Inter', sans-serif; font-size: 32px; font-weight: 400; color: rgba(255,255,255,0.7); margin-top: 16px; letter-spacing: 1px; }

/* ACCENT ELEMENTS */
.accent-line { width: 0; height: 4px; background: linear-gradient(90deg, #6366f1, #ec4899); margin: 30px auto 0; border-radius: 2px; }
@keyframes lineGrow { 0% { width: 0; } 100% { width: 120px; } }

/* PROGRESS BAR (top of screen) */
.progress-bar { position: absolute; top: 60px; left: 80px; right: 80px; height: 4px; background: rgba(255,255,255,0.2); border-radius: 2px; z-index: 100; display: flex; gap: 8px; }
.progress-segment { flex: 1; height: 100%; background: rgba(255,255,255,0.3); border-radius: 2px; overflow: hidden; }
.progress-segment.active::after { content: ''; display: block; height: 100%; background: white; animation: progressFill var(--duration, 3s) linear forwards; }
.progress-segment.done { background: white; }
@keyframes progressFill { 0% { width: 0; } 100% { width: 100%; } }

/* KEYFRAMES */
@keyframes floatIn {
  0% { opacity: 0; transform: scale(0.85) translateY(40px); }
  100% { opacity: 1; transform: scale(1) translateY(0); }
}
@keyframes float {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-12px); }
}
@keyframes fadeUp {
  0% { opacity: 0; transform: translateY(30px); }
  100% { opacity: 1; transform: translateY(0); }
}
@keyframes zoomIn {
  0% { opacity: 0; transform: scale(1.08); }
  100% { opacity: 1; transform: scale(1); }
}
</style>
```

PREMIUM ELEMENTS TO INCLUDE:
1. Add <div class="bg-glow"></div> as first child of reel-container (animated ambient glow)
2. Add <div class="accent-line"></div> after headlines for style
3. Use class="text-gradient" on key words in headlines for gradient text effect
4. Add progress bar at top showing video segments

IMAGE TREATMENT - CHOOSE BASED ON IMAGE TYPE:

**PRODUCT treatment** (isolated shots):
<div class="product-wrap"><img src="URL" class="product-img"></div>

**LIFESTYLE treatment** (contextual/environmental):
<div class="frame lifestyle active"><img src="URL" class="lifestyle-img"><div class="lifestyle-overlay"></div><div class="text-area">...</div></div>

FRAME STRUCTURE:
1. HERO: Impactful opening - lifestyle OR dramatic product reveal
2. FEATURE: Product detail + benefit (use accent-line)
3. VALUE: Social proof or key differentiator
4. CTA: Strong call-to-action with product

⚠️ INSTAGRAM SAFE ZONE RULES:
- Text at 15% from bottom (300px) - text-area class handles this
- Product images: max 750px tall, centered vertically in safe zone
- Keep right side clear: 200px padding on right for buttons
- Top 200px reserved for profile UI
- NEVER put important content in bottom 300px or right 180px

COMPOSITION BALANCE:
- Product should be vertically centered with breathing room
- Text below product with comfortable spacing (60px gap)
- Don't squash elements together - use the full safe zone height
- Distribute content evenly: product in upper area, text in lower area

Add at end: <script>const timing = [3500, 3500, 3500, 3500, 3500];</script>

Return ONLY complete HTML. No explanations."""

    # Build image list
    images_text = "\n".join([f"{i+1}. {img}" for i, img in enumerate(product_images)]) if product_images else "No images found - use placeholder styling"

    user_prompt = f"""Product: {page_title}
URL: {url}

IMAGES (use these exact URLs with class="product-img"):
{images_text}

{f"EXTRA: {prompt}" if prompt else ""}

Remember: Image ABOVE (900px wide, centered), text BELOW in text-area div. Dark background, NO white boxes."""

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

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Engine</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Inter', -apple-system, sans-serif; background: #0a0a0a; color: white; min-height: 100vh; padding: 60px 20px; }
        .container { max-width: 600px; margin: 0 auto; }
        h1 { font-size: 32px; font-weight: 700; margin-bottom: 8px; }
        .subtitle { color: #888; margin-bottom: 40px; }
        label { display: block; font-size: 14px; color: #888; margin-bottom: 8px; }
        input { width: 100%; padding: 16px; font-size: 16px; border: 1px solid #333; border-radius: 8px; background: #111; color: white; margin-bottom: 20px; }
        input:focus { outline: none; border-color: #6366f1; }
        button { width: 100%; padding: 16px; font-size: 16px; font-weight: 600; border: none; border-radius: 8px; background: linear-gradient(135deg, #6366f1, #ec4899); color: white; cursor: pointer; transition: transform 0.2s, opacity 0.2s; }
        button:hover { transform: translateY(-2px); }
        button:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
        .status { margin-top: 30px; padding: 20px; border-radius: 8px; background: #111; border: 1px solid #222; }
        .status.hidden { display: none; }
        .progress-bar { height: 4px; background: #333; border-radius: 2px; margin: 15px 0; overflow: hidden; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #6366f1, #ec4899); width: 0%; transition: width 0.3s; }
        .download-btn { display: inline-block; margin-top: 15px; padding: 12px 24px; background: #22c55e; border-radius: 6px; color: white; text-decoration: none; font-weight: 600; }
        .download-btn:hover { background: #16a34a; }
        .error { color: #ef4444; }
        .api-info { margin-top: 60px; padding-top: 30px; border-top: 1px solid #222; }
        .api-info h3 { font-size: 16px; margin-bottom: 15px; }
        code { background: #1a1a1a; padding: 2px 6px; border-radius: 4px; font-size: 13px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Video Engine</h1>
        <p class="subtitle">Generate premium product videos from any URL</p>

        <form id="generateForm">
            <label>Product URL</label>
            <input type="url" id="urlInput" placeholder="https://www.nike.com/t/air-max-90-mens-shoes" required>

            <label>Custom Instructions (optional)</label>
            <input type="text" id="promptInput" placeholder="Focus on comfort features...">

            <button type="submit" id="submitBtn">Generate Video</button>
        </form>

        <div id="status" class="status hidden">
            <div id="statusText">Starting...</div>
            <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
            <div id="downloadArea"></div>
        </div>

        <div class="api-info">
            <h3>API Endpoints</h3>
            <p><code>POST /generate-from-url</code> - Generate video from URL</p>
            <p><code>GET /status/{video_id}</code> - Check progress</p>
            <p><code>GET /download/{video_id}</code> - Download video</p>
        </div>
    </div>

    <script>
        const form = document.getElementById('generateForm');
        const statusDiv = document.getElementById('status');
        const statusText = document.getElementById('statusText');
        const progressFill = document.getElementById('progressFill');
        const downloadArea = document.getElementById('downloadArea');
        const submitBtn = document.getElementById('submitBtn');

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const url = document.getElementById('urlInput').value;
            const prompt = document.getElementById('promptInput').value;

            submitBtn.disabled = true;
            submitBtn.textContent = 'Generating...';
            statusDiv.classList.remove('hidden');
            statusText.textContent = 'Starting video generation...';
            progressFill.style.width = '0%';
            downloadArea.innerHTML = '';

            try {
                const res = await fetch('/generate-from-url', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url, prompt })
                });
                const data = await res.json();

                if (data.video_id) {
                    pollStatus(data.video_id);
                } else {
                    throw new Error(data.detail || 'Failed to start');
                }
            } catch (err) {
                statusText.innerHTML = '<span class="error">Error: ' + err.message + '</span>';
                submitBtn.disabled = false;
                submitBtn.textContent = 'Generate Video';
            }
        });

        async function pollStatus(videoId) {
            try {
                const res = await fetch('/status/' + videoId);
                const data = await res.json();

                progressFill.style.width = data.progress + '%';

                if (data.status === 'complete') {
                    statusText.textContent = 'Video ready!';
                    progressFill.style.width = '100%';
                    downloadArea.innerHTML = '<a href="/download/' + videoId + '" class="download-btn" download>Download Video</a>';
                    submitBtn.disabled = false;
                    submitBtn.textContent = 'Generate Another';
                } else if (data.status === 'error') {
                    statusText.innerHTML = '<span class="error">Error: ' + data.error + '</span>';
                    submitBtn.disabled = false;
                    submitBtn.textContent = 'Try Again';
                } else {
                    statusText.textContent = 'Generating... ' + data.status + ' (' + data.progress + '%)';
                    setTimeout(() => pollStatus(videoId), 1500);
                }
            } catch (err) {
                setTimeout(() => pollStatus(videoId), 2000);
            }
        }
    </script>
</body>
</html>
"""

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
