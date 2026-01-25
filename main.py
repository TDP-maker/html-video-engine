import os
import uuid
import asyncio
import subprocess
import imageio_ffmpeg
import httpx
import base64
import io
from fastapi import FastAPI, BackgroundTasks, HTTPException, Header, Depends
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from playwright.async_api import async_playwright
import uvicorn
from openai import OpenAI
import re
from datetime import datetime
from PIL import Image
from collections import Counter

# --- REMOVE.BG BACKGROUND REMOVAL ---
REMOVE_BG_API_KEY = os.environ.get("REMOVE_BG_API_KEY", "")

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

class VideoFeatures(BaseModel):
    """Toggleable premium features"""
    background_removal: bool = True      # Remove background from products (needs API key)
    ai_background: bool = True           # Generate AI cinematic backgrounds
    color_extraction: bool = True        # Extract brand colors from product
    cta_button: bool = True              # Animated CTA button on final frame
    progress_bar: bool = True            # Progress bar at top
    text_effects: bool = True            # Gradient text, accent lines
    floating_animation: bool = True      # Product float animation
    ken_burns: bool = True               # Slow zoom on lifestyle images
    smart_copy: bool = True              # Only use text from page (no hallucination)
    price_badge: bool = True             # Animated price badge with sale strike-through
    trust_badges: bool = True            # Trust badges (Free Shipping, Authentic, etc.)
    multi_format: bool = False           # Generate multiple formats (9:16, 1:1, 16:9)
    # Style customization
    font_family: str = "Inter"           # Font for headlines
    text_color: str = "#ffffff"          # Main text color
    accent_color: str = ""               # Accent color (empty = use extracted)

class URLVideoRequest(BaseModel):
    """Generate video from URL + prompt"""
    url: str
    prompt: str = ""
    format: str = "reel"
    fps: int = 30
    record_id: str = ""  # Airtable record ID for tracking
    features: VideoFeatures = VideoFeatures()  # Feature toggles

# --- FORMAT CONFIGURATIONS ---
FORMAT_DIMENSIONS = {
    "reel": (1080, 1920),      # 9:16 - Instagram Reels, TikTok, Stories
    "square": (1080, 1080),    # 1:1 - Instagram Feed, Facebook
    "landscape": (1920, 1080), # 16:9 - YouTube, Twitter, LinkedIn
}

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
    width, height = FORMAT_DIMENSIONS.get(format, FORMAT_DIMENSIONS["reel"])

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
                        # First slide - activate and let it stabilize
                        await page.evaluate("""() => {
                            document.querySelectorAll('.frame')[0].classList.add('active');
                        }""")
                        await page.wait_for_timeout(500)  # Wait for CSS to fully apply

                        # Capture frames for this slide
                        slide_frames = int((slide_duration_ms / 1000) * fps)
                        for i in range(slide_frames):
                            await page.wait_for_timeout(int(1000 / fps))  # Wait first for stable frame
                            await page.screenshot(path=f"{output_folder}/frame_{frame_index:04d}.png")
                            frame_index += 1
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

                        # Capture transition (1.5 second smooth crossfade)
                        transition_duration = 1.5
                        transition_frames = int(transition_duration * fps)
                        for i in range(transition_frames):
                            await page.wait_for_timeout(int(1000 / fps))  # Wait first, then capture
                            await page.screenshot(path=f"{output_folder}/frame_{frame_index:04d}.png")
                            frame_index += 1

                        # Clean up exit class
                        await page.evaluate(f"""(slideNum) => {{
                            document.querySelectorAll('.frame')[slideNum - 1].classList.remove('exit');
                        }}""", slide_num)

                        # Capture hold frames (remaining duration minus transition)
                        hold_frames = int((slide_duration_ms / 1000) * fps) - transition_frames
                        for i in range(max(0, hold_frames)):
                            await page.wait_for_timeout(int(1000 / fps))  # Wait first for stable frame
                            await page.screenshot(path=f"{output_folder}/frame_{frame_index:04d}.png")
                            frame_index += 1

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


async def extract_images_with_playwright(url: str) -> tuple[list, str, str, dict]:
    """Use Playwright to load page and extract images and text from rendered DOM"""
    product_images = []
    page_title = ""
    page_description = ""
    extracted_copy = {
        "product_name": "",
        "price": "",
        "description": "",
        "features": [],
        "brand": "",
        "cta_text": "",
        "all_text": []
    }

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

            # Extract ALL text content from the page for smart copy
            text_content = await page.evaluate("""() => {
                const result = {
                    product_name: '',
                    price: '',
                    description: '',
                    features: [],
                    brand: '',
                    cta_text: '',
                    headings: [],
                    paragraphs: []
                };

                // Product name - try multiple selectors
                const nameSelectors = [
                    'h1', '[data-testid="product-title"]', '.product-title', '.product-name',
                    '[itemprop="name"]', '.pdp-title', '#productTitle', '.product__title'
                ];
                for (const sel of nameSelectors) {
                    const el = document.querySelector(sel);
                    if (el && el.textContent.trim()) {
                        result.product_name = el.textContent.trim().substring(0, 200);
                        break;
                    }
                }

                // Price - try multiple selectors
                const priceSelectors = [
                    '[data-testid="price"]', '.price', '[itemprop="price"]', '.product-price',
                    '.current-price', '.sale-price', '#priceblock_ourprice', '.a-price-whole',
                    '[data-price]', '.pdp-price'
                ];
                for (const sel of priceSelectors) {
                    const el = document.querySelector(sel);
                    if (el && el.textContent.trim()) {
                        const priceText = el.textContent.trim();
                        // Look for price pattern
                        const priceMatch = priceText.match(/[$£€]?\\s*[\\d,]+\\.?\\d*/);
                        if (priceMatch) {
                            result.price = priceMatch[0].trim();
                            break;
                        }
                    }
                }

                // Description
                const descSelectors = [
                    '[itemprop="description"]', '.product-description', '.description',
                    '#productDescription', '.product-detail', '.pdp-description',
                    '[data-testid="product-description"]'
                ];
                for (const sel of descSelectors) {
                    const el = document.querySelector(sel);
                    if (el && el.textContent.trim()) {
                        result.description = el.textContent.trim().substring(0, 500);
                        break;
                    }
                }

                // Brand
                const brandSelectors = [
                    '[itemprop="brand"]', '.brand', '.product-brand', '[data-testid="brand"]',
                    '.manufacturer', '.vendor'
                ];
                for (const sel of brandSelectors) {
                    const el = document.querySelector(sel);
                    if (el && el.textContent.trim()) {
                        result.brand = el.textContent.trim().substring(0, 100);
                        break;
                    }
                }

                // Features/bullet points
                const featureSelectors = [
                    '.feature-list li', '.product-features li', '[data-testid="features"] li',
                    '.benefits li', '.highlights li', '#feature-bullets li',
                    '.product-highlights li', 'ul.features li'
                ];
                for (const sel of featureSelectors) {
                    document.querySelectorAll(sel).forEach(li => {
                        const text = li.textContent.trim();
                        if (text && text.length > 5 && text.length < 200) {
                            result.features.push(text);
                        }
                    });
                    if (result.features.length > 0) break;
                }

                // CTA buttons
                const ctaSelectors = [
                    'button[type="submit"]', '.add-to-cart', '.buy-now', '[data-testid="add-to-cart"]',
                    '.cta-button', '.purchase-button', '#add-to-cart-button'
                ];
                for (const sel of ctaSelectors) {
                    const el = document.querySelector(sel);
                    if (el && el.textContent.trim()) {
                        result.cta_text = el.textContent.trim().substring(0, 50);
                        break;
                    }
                }

                // All headings
                document.querySelectorAll('h1, h2, h3').forEach(h => {
                    const text = h.textContent.trim();
                    if (text && text.length > 3 && text.length < 150) {
                        result.headings.push(text);
                    }
                });
                result.headings = result.headings.slice(0, 10);

                // Key paragraphs (first few meaningful ones)
                document.querySelectorAll('p').forEach(p => {
                    const text = p.textContent.trim();
                    if (text && text.length > 30 && text.length < 300) {
                        result.paragraphs.push(text);
                    }
                });
                result.paragraphs = result.paragraphs.slice(0, 5);

                return result;
            }""")

            extracted_copy = text_content
            print(f"📝 Extracted copy: {extracted_copy.get('product_name', 'Unknown')[:50]}...")

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

    return product_images, page_title, page_description, extracted_copy


async def remove_background(image_url: str) -> str:
    """Remove background from product image using remove.bg API"""
    if not REMOVE_BG_API_KEY:
        print("⚠️ No REMOVE_BG_API_KEY set, skipping background removal")
        return image_url

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://api.remove.bg/v1.0/removebg",
                headers={"X-Api-Key": REMOVE_BG_API_KEY},
                data={
                    "image_url": image_url,
                    "size": "auto",
                    "format": "png"
                }
            )

            if response.status_code == 200:
                # Save the PNG with transparent background
                current_folder = os.getcwd()
                bg_removed_folder = os.path.join(current_folder, "bg_removed")
                os.makedirs(bg_removed_folder, exist_ok=True)

                filename = f"{uuid.uuid4().hex[:8]}.png"
                filepath = os.path.join(bg_removed_folder, filename)

                with open(filepath, "wb") as f:
                    f.write(response.content)

                print(f"✅ Background removed: {filename}")
                # Return as file:// URL for Playwright to load
                return f"file://{filepath}"
            else:
                print(f"⚠️ remove.bg error: {response.status_code} - {response.text}")
                return image_url

    except Exception as e:
        print(f"⚠️ Background removal failed: {e}")
        return image_url


async def process_images_for_premium(product_images: list) -> list:
    """Process images: remove backgrounds for product shots"""
    if not REMOVE_BG_API_KEY or not product_images:
        return product_images

    processed = []
    # Only process first 3 images to save API costs
    for i, img_url in enumerate(product_images[:3]):
        print(f"🎨 Processing image {i+1}/3...")
        processed_url = await remove_background(img_url)
        processed.append(processed_url)

    # Add remaining unprocessed images
    processed.extend(product_images[3:])
    return processed


def detect_product_category(title: str, description: str = "") -> str:
    """Detect product category from title and description"""
    text = (title + " " + description).lower()

    # Fashion/Clothing keywords
    fashion_keywords = ['dress', 'shirt', 'pants', 'jeans', 'jacket', 'coat', 'sweater',
                       'blouse', 'skirt', 'shorts', 'clothing', 'apparel', 'fashion',
                       'wear', 'outfit', 'style', 'collection', 'women', 'men', 'kids',
                       'top', 'bottom', 'suit', 'blazer', 'cardigan', 'hoodie', 't-shirt',
                       'everyday', 'casual', 'formal', 'loungewear', 'activewear']

    # Footwear keywords
    footwear_keywords = ['shoe', 'sneaker', 'boot', 'sandal', 'heel', 'loafer',
                        'trainer', 'footwear', 'nike', 'adidas', 'jordan', 'air max']

    # Beauty/Skincare keywords
    beauty_keywords = ['skincare', 'makeup', 'cosmetic', 'serum', 'cream', 'moisturizer',
                      'lipstick', 'foundation', 'beauty', 'skin', 'face', 'anti-aging']

    # Tech/Electronics keywords
    tech_keywords = ['phone', 'laptop', 'computer', 'tablet', 'headphone', 'speaker',
                    'watch', 'smart', 'wireless', 'bluetooth', 'electronic', 'device']

    # Food/Beverage keywords
    food_keywords = ['food', 'drink', 'beverage', 'coffee', 'tea', 'snack', 'organic',
                    'protein', 'supplement', 'vitamin', 'health']

    # Home/Furniture keywords
    home_keywords = ['furniture', 'home', 'decor', 'chair', 'table', 'sofa', 'lamp',
                    'bed', 'pillow', 'rug', 'curtain', 'kitchen']

    # Check categories (order matters - more specific first)
    if any(kw in text for kw in footwear_keywords):
        return "footwear"
    elif any(kw in text for kw in fashion_keywords):
        return "fashion"
    elif any(kw in text for kw in beauty_keywords):
        return "beauty"
    elif any(kw in text for kw in tech_keywords):
        return "tech"
    elif any(kw in text for kw in food_keywords):
        return "food"
    elif any(kw in text for kw in home_keywords):
        return "home"
    else:
        return "general"


async def generate_ai_background(product_title: str, product_category: str = "", brand_colors: dict = None) -> str:
    """Generate a context-appropriate background using DALL-E based on product category"""
    try:
        client = OpenAI()

        # Auto-detect category if not provided
        if not product_category:
            product_category = detect_product_category(product_title)

        print(f"🎨 Detected product category: {product_category}")

        # Use extracted brand colors if available (but subtly)
        if brand_colors and brand_colors.get("primary"):
            color_hint = f"Subtle accent lighting in {brand_colors['primary_name']} tones."
        else:
            color_hint = "Neutral, sophisticated lighting."

        # Category-specific background prompts - realistic, contextual environments
        bg_prompts = {
            "fashion": f"""Professional fashion photography studio backdrop.
Clean minimalist environment with soft diffused lighting.
Light gray seamless paper background or elegant white cyclorama wall.
Soft natural window light from the side, subtle shadows.
{color_hint}
Style: high-end fashion editorial, Vogue/Elle magazine aesthetic.
Clean, modern, sophisticated, premium feel.
NO people, NO clothing, NO mannequins, NO text, NO products.
ONLY the empty studio background environment.
Vertical 9:16 aspect ratio composition.""",

            "footwear": f"""Urban sneaker photography environment.
Clean concrete floor with subtle texture, industrial aesthetic.
Dramatic side lighting with soft shadows on polished surface.
Minimalist urban backdrop - clean walls, subtle textures.
{color_hint}
Style: Nike/Adidas campaign aesthetic, streetwear photography.
Moody but clean, premium athletic brand feel.
NO shoes, NO feet, NO products, NO text.
ONLY the empty backdrop environment.
Vertical 9:16 aspect ratio.""",

            "beauty": f"""Luxury skincare and beauty photography backdrop.
Soft, dreamy lighting with gentle gradients.
Elegant marble or soft fabric surface texture.
Warm, spa-like atmosphere with subtle rose gold or cream tones.
{color_hint}
Style: luxury cosmetics brand campaign, Glossier/La Mer aesthetic.
Soft focus areas, ethereal glow, premium serene feel.
NO products, NO bottles, NO text.
ONLY the empty background.
Vertical 9:16 aspect ratio.""",

            "tech": f"""Modern tech product photography environment.
Sleek dark surface with subtle reflections.
Clean gradient backdrop from dark gray to black.
Soft blue or white accent lighting, futuristic feel.
{color_hint}
Style: Apple product photography aesthetic, minimal and premium.
Clean lines, subtle light rays, polished reflective surface.
NO devices, NO electronics, NO text, NO products.
ONLY the empty backdrop.
Vertical 9:16 aspect ratio.""",

            "food": f"""Appetizing food photography setting.
Natural wooden table or marble countertop surface.
Warm, inviting natural window light from the side.
Rustic-modern kitchen environment feel.
{color_hint}
Style: gourmet food magazine, Bon Appetit aesthetic.
Warm tones, soft shadows, appetizing atmosphere.
NO food, NO dishes, NO utensils, NO text.
ONLY the empty surface and background.
Vertical 9:16 aspect ratio.""",

            "home": f"""Interior design photography backdrop.
Elegant modern living space with soft natural light.
Neutral walls with subtle texture, warm ambient glow.
Clean architectural lines, sophisticated atmosphere.
{color_hint}
Style: Architectural Digest aesthetic, modern luxury interior.
Soft window light, clean minimalist space.
NO furniture, NO decor items, NO text.
ONLY the empty room environment.
Vertical 9:16 aspect ratio.""",

            "general": f"""Premium product photography backdrop.
Clean, sophisticated studio environment.
Soft gradient lighting from light gray to white.
Subtle shadows, professional commercial photography feel.
{color_hint}
Style: high-end advertising photography, clean and minimal.
Professional studio lighting, elegant simplicity.
NO products, NO text, NO logos, NO objects.
ONLY the empty background.
Vertical 9:16 aspect ratio."""
        }

        bg_prompt = bg_prompts.get(product_category, bg_prompts["general"])

        print(f"🎨 Generating {product_category} AI background...")

        response = client.images.generate(
            model="dall-e-3",
            prompt=bg_prompt,
            size="1024x1792",  # Vertical for reels
            quality="standard",
            n=1
        )

        bg_url = response.data[0].url
        print(f"✅ AI background generated")

        # Download and save locally
        async with httpx.AsyncClient(timeout=60) as http_client:
            img_response = await http_client.get(bg_url)
            if img_response.status_code == 200:
                current_folder = os.getcwd()
                bg_folder = os.path.join(current_folder, "ai_backgrounds")
                os.makedirs(bg_folder, exist_ok=True)

                filename = f"bg_{uuid.uuid4().hex[:8]}.png"
                filepath = os.path.join(bg_folder, filename)

                with open(filepath, "wb") as f:
                    f.write(img_response.content)

                return f"file://{filepath}"

        return bg_url

    except Exception as e:
        print(f"⚠️ AI background generation failed: {e}")
        return ""


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB to hex color"""
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def get_color_name(hex_color: str) -> str:
    """Get a descriptive name for a color based on its hue"""
    r, g, b = hex_to_rgb(hex_color)

    # Calculate brightness
    brightness = (r * 299 + g * 587 + b * 114) / 1000

    # Determine if it's a neutral color
    max_rgb = max(r, g, b)
    min_rgb = min(r, g, b)
    saturation = (max_rgb - min_rgb) / max_rgb if max_rgb > 0 else 0

    if saturation < 0.15:
        if brightness < 50:
            return "deep black"
        elif brightness < 128:
            return "dark gray"
        elif brightness < 200:
            return "light gray"
        else:
            return "white"

    # Calculate hue
    if max_rgb == min_rgb:
        hue = 0
    elif max_rgb == r:
        hue = 60 * ((g - b) / (max_rgb - min_rgb)) % 360
    elif max_rgb == g:
        hue = 60 * ((b - r) / (max_rgb - min_rgb)) + 120
    else:
        hue = 60 * ((r - g) / (max_rgb - min_rgb)) + 240

    # Map hue to color name
    if hue < 15 or hue >= 345:
        base = "red"
    elif hue < 45:
        base = "orange"
    elif hue < 75:
        base = "yellow"
    elif hue < 150:
        base = "green"
    elif hue < 195:
        base = "cyan"
    elif hue < 255:
        base = "blue"
    elif hue < 285:
        base = "purple"
    elif hue < 345:
        base = "pink"
    else:
        base = "red"

    # Add brightness modifier
    if brightness < 80:
        return f"deep {base}"
    elif brightness > 180:
        return f"light {base}"
    else:
        return base


def adjust_color_for_dark_bg(hex_color: str, min_brightness: int = 120) -> str:
    """Adjust a color to ensure visibility on dark backgrounds"""
    r, g, b = hex_to_rgb(hex_color)
    brightness = (r * 299 + g * 587 + b * 114) / 1000

    if brightness < min_brightness:
        # Lighten the color while preserving hue
        factor = min_brightness / max(brightness, 1)
        r = min(255, int(r * factor))
        g = min(255, int(g * factor))
        b = min(255, int(b * factor))

    return rgb_to_hex(r, g, b)


async def extract_colors_from_image(image_url: str) -> dict:
    """Extract dominant colors from a product image using k-means clustering"""
    try:
        print(f"🎨 Extracting colors from image...")

        # Download image
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(image_url)
            if response.status_code != 200:
                print(f"⚠️ Failed to download image for color extraction")
                return None

            image_data = response.content

        # Open and process image
        img = Image.open(io.BytesIO(image_data))
        img = img.convert('RGB')

        # Resize for faster processing
        img.thumbnail((150, 150))

        # Get all pixels
        pixels = list(img.getdata())

        # Filter out near-white and near-black pixels (background)
        filtered_pixels = []
        for r, g, b in pixels:
            brightness = (r * 299 + g * 587 + b * 114) / 1000
            saturation = (max(r, g, b) - min(r, g, b)) / max(max(r, g, b), 1)

            # Skip very dark, very light, or very unsaturated pixels
            if 30 < brightness < 240 and saturation > 0.1:
                # Quantize to reduce color variations (round to nearest 16)
                r = (r // 16) * 16
                g = (g // 16) * 16
                b = (b // 16) * 16
                filtered_pixels.append((r, g, b))

        if len(filtered_pixels) < 10:
            # Not enough colorful pixels, use all pixels
            filtered_pixels = [(
                (r // 16) * 16,
                (g // 16) * 16,
                (b // 16) * 16
            ) for r, g, b in pixels]

        # Count pixel frequencies
        color_counts = Counter(filtered_pixels)

        # Get top colors
        top_colors = color_counts.most_common(10)

        if not top_colors:
            print(f"⚠️ No colors extracted")
            return None

        # Select distinct colors (not too similar to each other)
        selected_colors = []
        for color, count in top_colors:
            if len(selected_colors) >= 3:
                break

            # Check if this color is distinct from already selected
            is_distinct = True
            for selected in selected_colors:
                # Calculate color distance
                dist = sum(abs(a - b) for a, b in zip(color, selected))
                if dist < 100:  # Too similar
                    is_distinct = False
                    break

            if is_distinct:
                selected_colors.append(color)

        # Ensure we have at least 3 colors (use variations if needed)
        while len(selected_colors) < 3:
            if selected_colors:
                # Create a variation of the first color
                r, g, b = selected_colors[0]
                if len(selected_colors) == 1:
                    # Complementary-ish color
                    selected_colors.append((255 - r, 255 - g, 255 - b))
                else:
                    # Shifted hue
                    selected_colors.append((g, b, r))
            else:
                # Fallback to warm gold theme (matches default colors)
                selected_colors = [(201, 169, 110), (139, 115, 85), (212, 175, 55)]
                break

        # Convert to hex and adjust for dark backgrounds
        primary_hex = rgb_to_hex(*selected_colors[0])
        secondary_hex = rgb_to_hex(*selected_colors[1])
        accent_hex = rgb_to_hex(*selected_colors[2])

        # Check if colors are too neutral/boring (all grays or near-white)
        def is_neutral(rgb):
            r, g, b = rgb
            max_rgb = max(r, g, b)
            min_rgb = min(r, g, b)
            saturation = (max_rgb - min_rgb) / max(max_rgb, 1)
            return saturation < 0.15  # Very low saturation = gray/white

        if all(is_neutral(c) for c in selected_colors):
            print(f"⚠️ Extracted colors too neutral, using warm gold fallback")
            return None  # Will trigger default warm gold theme

        # Adjust colors for visibility on dark background
        primary_adjusted = adjust_color_for_dark_bg(primary_hex)
        secondary_adjusted = adjust_color_for_dark_bg(secondary_hex)
        accent_adjusted = adjust_color_for_dark_bg(accent_hex)

        result = {
            "primary": primary_adjusted,
            "secondary": secondary_adjusted,
            "accent": accent_adjusted,
            "primary_name": get_color_name(primary_adjusted),
            "secondary_name": get_color_name(secondary_adjusted),
            "accent_name": get_color_name(accent_adjusted),
            "original_primary": primary_hex,
            "original_secondary": secondary_hex,
            "original_accent": accent_hex
        }

        print(f"✅ Extracted colors: {result['primary_name']} ({result['primary']}), {result['secondary_name']} ({result['secondary']}), {result['accent_name']} ({result['accent']})")

        return result

    except Exception as e:
        print(f"⚠️ Color extraction failed: {e}")
        return None


async def generate_html_from_url(url: str, prompt: str = "", features: VideoFeatures = None) -> str:
    """Use Playwright + OpenAI to generate HTML video content from a URL"""

    # Use default features if none provided
    if features is None:
        features = VideoFeatures()

    print(f"🔍 Extracting images and copy with Playwright...")
    print(f"⚙️ Features: BG Removal={features.background_removal}, AI BG={features.ai_background}, Colors={features.color_extraction}")
    product_images, page_title, page_description, extracted_copy = await extract_images_with_playwright(url)
    print(f"📸 Found {len(product_images)} product images")
    print(f"📝 Extracted product name: {extracted_copy.get('product_name', 'Unknown')[:50]}")

    # Detect product category early - this affects how we process images
    product_category = detect_product_category(page_title, page_description)
    print(f"🏷️ Detected category: {product_category}")

    # Categories where we should KEEP original images (lifestyle shots)
    # These look better with context/environment, not as cutouts
    lifestyle_categories = ["fashion", "home", "food"]
    is_lifestyle_product = product_category in lifestyle_categories

    # Premium feature: Extract brand colors from product image (if enabled)
    brand_colors = None
    if features.color_extraction and product_images:
        brand_colors = await extract_colors_from_image(product_images[0])

    # Premium feature: Remove backgrounds - BUT skip for lifestyle categories
    # Fashion/clothing looks terrible as floating cutouts - keep the context!
    if features.background_removal and REMOVE_BG_API_KEY and product_images:
        if is_lifestyle_product:
            print(f"⏭️ Skipping background removal for {product_category} (lifestyle images work better with context)")
        else:
            print(f"🎨 Removing backgrounds (premium mode)...")
            product_images = await process_images_for_premium(product_images)
            print(f"✅ Background removal complete")

    # Premium feature: Generate AI background with brand colors (if enabled)
    # For lifestyle categories, we might skip AI backgrounds since images have their own context
    ai_background_url = ""
    if features.ai_background:
        if is_lifestyle_product:
            print(f"⏭️ Skipping AI background for {product_category} (using original lifestyle images)")
        else:
            try:
                ai_background_url = await generate_ai_background(page_title, product_category, brand_colors)
            except Exception as e:
                print(f"⚠️ Skipping AI background: {e}")

    # Build smart copy constraints from extracted content (if enabled)
    smart_copy = []
    if features.smart_copy:
        if extracted_copy.get("product_name"):
            smart_copy.append(f"PRODUCT NAME: {extracted_copy['product_name']}")
        if extracted_copy.get("brand"):
            smart_copy.append(f"BRAND: {extracted_copy['brand']}")
        if extracted_copy.get("price"):
            smart_copy.append(f"PRICE: {extracted_copy['price']}")
        if extracted_copy.get("description"):
            smart_copy.append(f"DESCRIPTION: {extracted_copy['description'][:300]}")
        if extracted_copy.get("features"):
            features_text = " | ".join(extracted_copy['features'][:5])
            smart_copy.append(f"FEATURES: {features_text}")
        if extracted_copy.get("cta_text"):
            smart_copy.append(f"CTA TEXT: {extracted_copy['cta_text']}")
        if extracted_copy.get("headings"):
            headings_text = " | ".join(extracted_copy['headings'][:5])
            smart_copy.append(f"HEADINGS: {headings_text}")

    smart_copy_text = "\n".join(smart_copy) if smart_copy else "No specific copy extracted - use generic premium messaging"

    # Build enabled features list for logging
    enabled_features = []
    if features.background_removal: enabled_features.append("BG Removal")
    if features.ai_background: enabled_features.append("AI Background")
    if features.color_extraction: enabled_features.append("Color Extraction")
    if features.cta_button: enabled_features.append("CTA Button")
    if features.progress_bar: enabled_features.append("Progress Bar")
    if features.text_effects: enabled_features.append("Text Effects")
    if features.floating_animation: enabled_features.append("Float Animation")
    if features.ken_burns: enabled_features.append("Ken Burns")
    if features.smart_copy: enabled_features.append("Smart Copy")
    if features.price_badge: enabled_features.append("Price Badge")
    if features.trust_badges: enabled_features.append("Trust Badges")
    if features.multi_format: enabled_features.append("Multi-Format")
    print(f"⚙️ Enabled features: {', '.join(enabled_features)}")

    client = OpenAI()

    # Set up dynamic colors - use extracted brand colors or defaults
    if brand_colors:
        primary_color = brand_colors["primary"]
        secondary_color = brand_colors["secondary"]
        accent_color = brand_colors["accent"]
        primary_rgb = hex_to_rgb(primary_color)
        secondary_rgb = hex_to_rgb(secondary_color)
        accent_rgb = hex_to_rgb(accent_color)
        color_description = f"Brand colors extracted: {brand_colors['primary_name']} ({primary_color}), {brand_colors['secondary_name']} ({secondary_color}), {brand_colors['accent_name']} ({accent_color})"
    else:
        # Default neutral/warm theme - works for any product
        primary_color = "#c9a96e"    # Warm gold
        secondary_color = "#8b7355"  # Warm bronze
        accent_color = "#d4af37"     # Classic gold
        primary_rgb = (201, 169, 110)
        secondary_rgb = (139, 115, 85)
        accent_rgb = (212, 175, 55)
        color_description = "Using neutral warm gold theme (no brand colors extracted)"

    # Override with custom accent color if provided
    if features.accent_color and features.accent_color != "#c9a96e":
        accent_color = features.accent_color
        accent_rgb = tuple(int(features.accent_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        print(f"🎨 Using custom accent color: {accent_color}")

    # Custom text color
    text_color = features.text_color if features.text_color else "#ffffff"

    # Custom font
    font_family = features.font_family if features.font_family else "Inter"
    # Build Google Fonts URL with all weights
    font_url_name = font_family.replace(' ', '+')

    print(f"🎨 {color_description}")
    print(f"🔤 Font: {font_family}, Text: {text_color}, Accent: {accent_color}")

    system_prompt = f"""You are a premium video ad designer. Create cinematic Instagram Reel HTML videos.

⚠️ INSTAGRAM SAFE ZONES - CRITICAL:
- TOP 250px: Username/follow button overlay - NO important content
- BOTTOM 400px: Captions/music/buttons overlay - NO important text here
- RIGHT 150px: Like/comment/share buttons - keep content left of this
- SAFE AREA: Content should be within x:0-930px, y:250-1520px

🎨 BRAND COLORS (USE THESE THROUGHOUT):
- Primary: {primary_color} (for main glows, shadows)
- Secondary: {secondary_color} (for accent glows, gradients)
- Accent: {accent_color} (for text gradients, highlights)

MANDATORY STRUCTURE (copy this exactly, using the brand colors):
```
<style>
@import url('https://fonts.googleapis.com/css2?family={font_url_name}:wght@400;700;900&display=swap');
* {{ margin: 0; padding: 0; box-sizing: border-box; }}

/* ANIMATED GRADIENT BACKGROUND - Using brand colors */
body {{ background: #0a0a0a; }}
.reel-container {{
  width: 1080px; height: 1920px; position: relative; overflow: hidden;
  background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #0a0a0a 100%);
}}

/* CINEMATIC VIGNETTE - Darker edges like film */
.vignette {{
  position: absolute; top: 0; left: 0; width: 100%; height: 100%;
  background: radial-gradient(ellipse at center, transparent 40%, rgba(0,0,0,0.5) 100%);
  z-index: 50; pointer-events: none;
}}

/* FILM GRAIN OVERLAY - Subtle texture */
.film-grain {{
  position: absolute; top: 0; left: 0; width: 100%; height: 100%;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E");
  opacity: 0.03; z-index: 51; pointer-events: none; mix-blend-mode: overlay;
}}

/* CINEMATIC COLOR GRADE */
.color-grade {{
  position: absolute; top: 0; left: 0; width: 100%; height: 100%;
  background: linear-gradient(180deg, rgba(255,200,150,0.03) 0%, transparent 50%, rgba(100,150,255,0.03) 100%);
  z-index: 52; pointer-events: none; mix-blend-mode: color;
}}
.bg-glow {{
  position: absolute; width: 150%; height: 150%; top: -25%; left: -25%;
  background: radial-gradient(circle at 30% 20%, rgba({primary_rgb[0]},{primary_rgb[1]},{primary_rgb[2]},0.15) 0%, transparent 50%),
              radial-gradient(circle at 70% 80%, rgba({secondary_rgb[0]},{secondary_rgb[1]},{secondary_rgb[2]},0.1) 0%, transparent 50%);
  animation: glowMove 8s ease-in-out infinite;
}}
@keyframes glowMove {{
  0%, 100% {{ transform: translate(0, 0) scale(1); }}
  50% {{ transform: translate(10px, -10px) scale(1.02); }}
}}

/* FRAME TRANSITIONS - Smooth crossfade */
.frame {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; opacity: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 0; transition: opacity 1.5s ease-in-out; }}
.frame.active {{ opacity: 1; transition: opacity 1.5s ease-in-out; }}
.frame.exit {{ opacity: 0; transition: opacity 1.5s ease-in-out; }}

/* Safe zone helper - balanced vertical distribution */
.safe-zone {{ position: absolute; top: 200px; left: 80px; right: 180px; bottom: 350px; display: flex; flex-direction: column; align-items: center; justify-content: space-between; }}
.content-area {{ flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; width: 100%; }}

/* PRODUCT ANIMATIONS - Only for non-lifestyle */
.frame.active .product-wrap {{ animation: floatIn 1.2s ease-out forwards, float 6s ease-in-out 1.5s infinite; }}
.frame.active .text-area {{ animation: fadeUp 0.8s ease-out 0.4s forwards; opacity: 0; }}
.frame.active .accent-line {{ animation: lineGrow 0.6s ease-out 0.6s forwards; }}

/* LIFESTYLE - Override: NO animations at all */
.frame.lifestyle.active .lifestyle-img {{ animation: none !important; transform: none !important; }}
.frame.lifestyle.active .text-area {{ animation: fadeUp 0.8s ease-out 0.2s forwards; opacity: 0; }}

/* PRODUCT TREATMENT - Premium floating product with brand-colored glow */
.product-wrap {{ position: relative; transform: scale(0.9) translateY(20px); opacity: 0; z-index: 1; }}
.product-wrap::before {{
  content: ''; position: absolute; top: 50%; left: 50%;
  transform: translate(-50%, -50%); width: 120%; height: 120%;
  background: radial-gradient(ellipse at center, rgba({primary_rgb[0]},{primary_rgb[1]},{primary_rgb[2]},0.2) 0%, rgba({secondary_rgb[0]},{secondary_rgb[1]},{secondary_rgb[2]},0.1) 40%, transparent 70%);
  z-index: -1; border-radius: 50%;
  filter: blur(40px);
}}
.product-img {{ width: 950px; height: auto; max-height: 1200px; object-fit: contain; filter: drop-shadow(0 30px 60px rgba(0,0,0,0.5)) drop-shadow(0 0 100px rgba({primary_rgb[0]},{primary_rgb[1]},{primary_rgb[2]},0.3)); }}

/* LIFESTYLE TREATMENT - Full bleed, edge to edge, NO ANIMATION */
.frame.lifestyle {{ padding: 0 !important; background: #000; }}
.frame.lifestyle .bg-glow {{ display: none; }}  /* Hide animated glow for lifestyle */
.lifestyle-img {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; opacity: 1; z-index: 0; }}
.frame.active .lifestyle-img {{ animation: none; opacity: 1; }}  /* NO animation, just static */
.lifestyle-overlay {{ position: absolute; bottom: 0; left: 0; width: 100%; height: 50%; background: linear-gradient(to top, rgba(0,0,0,0.8) 0%, rgba(0,0,0,0.4) 50%, transparent 100%); z-index: 1; }}

/* LIFESTYLE TEXT - Only show ONE headline, hide extras */
.frame.lifestyle .text-area {{ z-index: 10; }}
.frame.lifestyle .text-area h1:not(:first-of-type) {{ display: none; }}  /* Hide extra h1s */
.frame.lifestyle .text-area p:not(:first-of-type) {{ display: none; }}  /* Hide extra paragraphs */
.frame.lifestyle .trust-badges {{ display: none; }}  /* No trust badges */
.frame.lifestyle .price-badge {{ display: none; }}  /* No price badges */

/* AI BACKGROUND TREATMENT - Cinematic atmosphere */
.ai-bg {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; opacity: 0.6; z-index: 0; }}
.frame.ai-background .ai-bg {{ animation: bgPulse 6s ease-in-out infinite; }}
@keyframes bgPulse {{ 0%, 100% {{ transform: scale(1); opacity: 0.6; }} 50% {{ transform: scale(1.05); opacity: 0.7; }} }}

/* PREMIUM TEXT STYLING - 15% from bottom (288px), above Instagram UI */
.text-area {{ position: absolute; bottom: 300px; left: 0; text-align: center; width: 100%; padding: 0 120px; padding-right: 200px; transform: translateY(30px); z-index: 10; }}
h1 {{
  font-family: '{font_family}', sans-serif; font-size: 72px; font-weight: 900;
  color: {text_color}; text-transform: uppercase; line-height: 1.1; letter-spacing: -1px;
  text-shadow: 0 4px 30px rgba(0,0,0,0.5), 0 0 60px rgba({primary_rgb[0]},{primary_rgb[1]},{primary_rgb[2]},0.3);
}}
/* Gradient text using brand colors */
.text-gradient {{
  background: linear-gradient(135deg, #fff 0%, {accent_color} 50%, #fff 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
}}
/* Bold gradient - stronger brand color presence */
.text-gradient-bold {{
  background: linear-gradient(135deg, {primary_color} 0%, {accent_color} 50%, {secondary_color} 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
}}
/* Brand-colored headline */
.text-brand {{
  color: {primary_color};
  text-shadow: 0 4px 30px rgba(0,0,0,0.5), 0 0 40px rgba({primary_rgb[0]},{primary_rgb[1]},{primary_rgb[2]},0.4);
}}
/* Accent colored text */
.text-accent {{
  color: {accent_color};
}}
p {{ font-family: '{font_family}', sans-serif; font-size: 32px; font-weight: 400; color: {text_color}; opacity: 0.7; margin-top: 16px; letter-spacing: 1px; }}
/* Subtitle with brand color */
p.subtitle-brand {{
  color: {primary_color};
  opacity: 0.9;
}}
/* Highlighted keywords */
.highlight {{
  color: {accent_color};
  font-weight: 700;
}}

/* ACCENT ELEMENTS - Using brand colors */
.accent-line {{ width: 0; height: 4px; background: linear-gradient(90deg, {primary_color}, {secondary_color}); margin: 30px auto 0; border-radius: 2px; }}
@keyframes lineGrow {{ 0% {{ width: 0; }} 100% {{ width: 120px; }} }}

/* CTA BUTTON - Animated call-to-action */
.cta-button {{
  display: inline-block;
  padding: 20px 50px;
  font-family: 'Inter', sans-serif;
  font-size: 28px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 2px;
  color: white;
  background: linear-gradient(135deg, {primary_color}, {secondary_color});
  border: none;
  border-radius: 50px;
  cursor: pointer;
  position: relative;
  overflow: hidden;
  box-shadow: 0 10px 40px rgba({primary_rgb[0]},{primary_rgb[1]},{primary_rgb[2]},0.4), 0 0 60px rgba({primary_rgb[0]},{primary_rgb[1]},{primary_rgb[2]},0.2);
  opacity: 0;
  transform: scale(0.8) translateY(30px);
}}
.cta-button::before {{
  content: '';
  position: absolute;
  top: 0; left: -100%;
  width: 100%; height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
  transition: left 0.5s;
}}
.cta-button::after {{
  content: '';
  position: absolute;
  inset: -3px;
  background: linear-gradient(135deg, {primary_color}, {secondary_color}, {accent_color});
  border-radius: 50px;
  z-index: -1;
  opacity: 0;
  animation: ctaPulse 2s ease-in-out infinite;
}}
.frame.active .cta-button {{
  animation: ctaAppear 0.8s cubic-bezier(0.34, 1.56, 0.64, 1) 0.6s forwards, ctaFloat 3s ease-in-out 1.4s infinite;
}}
.frame.active .cta-button::before {{
  animation: ctaShine 2s ease-in-out 1s infinite;
}}
@keyframes ctaAppear {{
  0% {{ opacity: 0; transform: scale(0.8) translateY(30px); }}
  100% {{ opacity: 1; transform: scale(1) translateY(0); }}
}}
@keyframes ctaFloat {{
  0%, 100% {{ transform: translateY(0); }}
  50% {{ transform: translateY(-8px); }}
}}
@keyframes ctaShine {{
  0% {{ left: -100%; }}
  50%, 100% {{ left: 100%; }}
}}
@keyframes ctaPulse {{
  0%, 100% {{ opacity: 0; transform: scale(1); }}
  50% {{ opacity: 0.5; transform: scale(1.05); }}
}}

/* PRICE BADGE - Animated price tag with sale support */
.price-badge {{
  display: inline-flex;
  flex-direction: column;
  align-items: center;
  padding: 15px 30px;
  background: linear-gradient(135deg, rgba(0,0,0,0.8), rgba(20,20,20,0.9));
  border: 2px solid {primary_color};
  border-radius: 12px;
  position: relative;
  overflow: hidden;
  opacity: 0;
  transform: scale(0.8) rotate(-3deg);
  box-shadow: 0 10px 40px rgba(0,0,0,0.5), 0 0 30px rgba({primary_rgb[0]},{primary_rgb[1]},{primary_rgb[2]},0.2);
}}
.price-badge::before {{
  content: '';
  position: absolute;
  top: 0; left: -100%;
  width: 100%; height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
}}
.frame.active .price-badge {{
  animation: priceAppear 0.6s cubic-bezier(0.34, 1.56, 0.64, 1) 0.8s forwards;
}}
.frame.active .price-badge::before {{
  animation: priceShine 2s ease-in-out 1.2s infinite;
}}
.price-current {{
  font-family: 'Inter', sans-serif;
  font-size: 48px;
  font-weight: 900;
  color: white;
  text-shadow: 0 2px 10px rgba(0,0,0,0.5);
}}
.price-original {{
  font-family: 'Inter', sans-serif;
  font-size: 24px;
  font-weight: 400;
  color: rgba(255,255,255,0.5);
  text-decoration: line-through;
  margin-bottom: 5px;
}}
.price-discount {{
  position: absolute;
  top: -10px;
  right: -10px;
  background: linear-gradient(135deg, {secondary_color}, {primary_color});
  color: white;
  font-family: 'Inter', sans-serif;
  font-size: 14px;
  font-weight: 700;
  padding: 8px 12px;
  border-radius: 20px;
  transform: rotate(10deg);
  animation: discountPulse 1.5s ease-in-out infinite;
}}
@keyframes priceAppear {{
  0% {{ opacity: 0; transform: scale(0.8) rotate(-3deg); }}
  100% {{ opacity: 1; transform: scale(1) rotate(0deg); }}
}}
@keyframes priceShine {{
  0% {{ left: -100%; }}
  50%, 100% {{ left: 100%; }}
}}
@keyframes discountPulse {{
  0%, 100% {{ transform: rotate(10deg) scale(1); }}
  50% {{ transform: rotate(10deg) scale(1.1); }}
}}

/* TRUST BADGES - Animated trust indicators */
.trust-badges {{
  display: flex;
  gap: 15px;
  justify-content: center;
  flex-wrap: wrap;
  margin-top: 20px;
  opacity: 0;
  transform: translateY(20px);
}}
.frame.active .trust-badges {{
  animation: trustAppear 0.6s ease-out 1s forwards;
}}
.trust-badge {{
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 16px;
  background: rgba(255,255,255,0.1);
  border: 1px solid rgba(255,255,255,0.2);
  border-radius: 25px;
  backdrop-filter: blur(10px);
}}
.trust-badge .icon {{
  font-size: 18px;
}}
.trust-badge .text {{
  font-family: 'Inter', sans-serif;
  font-size: 13px;
  font-weight: 600;
  color: white;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}}
@keyframes trustAppear {{
  0% {{ opacity: 0; transform: translateY(20px); }}
  100% {{ opacity: 1; transform: translateY(0); }}
}}

/* PROGRESS BAR (top of screen) */
.progress-bar {{ position: absolute; top: 60px; left: 80px; right: 80px; height: 4px; background: rgba(255,255,255,0.2); border-radius: 2px; z-index: 100; display: flex; gap: 8px; }}
.progress-segment {{ flex: 1; height: 100%; background: rgba(255,255,255,0.3); border-radius: 2px; overflow: hidden; }}
.progress-segment.active::after {{ content: ''; display: block; height: 100%; background: white; animation: progressFill var(--duration, 3s) linear forwards; }}
.progress-segment.done {{ background: white; }}
@keyframes progressFill {{ 0% {{ width: 0; }} 100% {{ width: 100%; }} }}

/* KEYFRAMES */
@keyframes floatIn {{
  0% {{ opacity: 0; transform: scale(0.85) translateY(40px); }}
  100% {{ opacity: 1; transform: scale(1) translateY(0); }}
}}
@keyframes float {{
  0%, 100% {{ transform: translateY(0); }}
  50% {{ transform: translateY(-6px); }}
}}
@keyframes fadeUp {{
  0% {{ opacity: 0; transform: translateY(40px); filter: blur(4px); }}
  100% {{ opacity: 1; transform: translateY(0); filter: blur(0); }}
}}
@keyframes zoomIn {{
  0% {{ opacity: 0; transform: scale(1.03); }}
  100% {{ opacity: 1; transform: scale(1); }}
}}
/* CINEMATIC TEXT REVEAL - Slide up with mask */
@keyframes slideReveal {{
  0% {{ clip-path: inset(100% 0 0 0); transform: translateY(20px); }}
  100% {{ clip-path: inset(0 0 0 0); transform: translateY(0); }}
}}
.text-reveal {{ animation: slideReveal 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94) forwards; }}

/* LIGHT LEAK - Subtle flare on transitions */
@keyframes lightLeak {{
  0% {{ opacity: 0; transform: translateX(-100%); }}
  50% {{ opacity: 0.3; }}
  100% {{ opacity: 0; transform: translateX(100%); }}
}}
.light-leak {{
  position: absolute; top: 0; left: 0; width: 100%; height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255,200,150,0.4), rgba(255,255,255,0.2), transparent);
  z-index: 45; pointer-events: none; opacity: 0;
}}
.frame.active .light-leak {{ animation: lightLeak 1.5s ease-out forwards; }}
</style>
```

PREMIUM ELEMENTS TO INCLUDE:
1. Add cinematic overlays as LAST children of reel-container:
   <div class="vignette"></div>
   <div class="film-grain"></div>
   <div class="color-grade"></div>
{"2. Add <div class='accent-line'></div> after headlines for style (uses brand gradient)" if features.text_effects else ""}
{"3. TEXT STYLING WITH BRAND COLORS - use these classes to match the product:" if features.text_effects else ""}
{"   - class='text-gradient' - subtle gradient (white to brand accent)" if features.text_effects else ""}
{"   - class='text-gradient-bold' - full brand color gradient (primary → accent → secondary)" if features.text_effects else ""}
{"   - class='text-brand' - solid brand primary color with glow" if features.text_effects else ""}
{"   - class='text-accent' - brand accent color" if features.text_effects else ""}
{"   - class='highlight' - accent color for keywords within text" if features.text_effects else ""}
{"   - p.subtitle-brand - subtitles in brand color" if features.text_effects else ""}
{"   USE THESE to make text match the product's extracted colors!" if features.text_effects else ""}
{"4. Add progress bar at top showing video segments" if features.progress_bar else ""}
{"5. Add <button class='cta-button'>SHOP NOW</button> on the FINAL frame (animated CTA)" if features.cta_button else ""}
{"6. Add PRICE BADGE when price is available - use this HTML structure:" if features.price_badge else ""}
{'''   <div class="price-badge">
     <span class="price-original">$XX.XX</span>  <!-- Only if sale/original price exists -->
     <span class="price-current">$XX.XX</span>
     <span class="price-discount">XX% OFF</span>  <!-- Only if discount exists -->
   </div>''' if features.price_badge else ""}
{"7. Add TRUST BADGES on hero or CTA frame - use this HTML structure:" if features.trust_badges else ""}
{'''   <div class="trust-badges">
     <div class="trust-badge"><span class="icon">🚚</span><span class="text">Free Shipping</span></div>
     <div class="trust-badge"><span class="icon">✓</span><span class="text">100% Authentic</span></div>
     <div class="trust-badge"><span class="icon">⚡</span><span class="text">Fast Delivery</span></div>
   </div>''' if features.trust_badges else ""}

IMAGE TREATMENT - CHOOSE BASED ON IMAGE TYPE:

**PRODUCT treatment** (isolated shots):
<div class="product-wrap"><img src="URL" class="product-img"></div>

{"**PRODUCT + AI BACKGROUND** (premium cinematic look):" if features.ai_background else ""}
{'''<div class="frame ai-background active">
  <img src="AI_BG_URL" class="ai-bg">
  <div class="product-wrap"><img src="URL" class="product-img"></div>
  <div class="text-area">...</div>
</div>''' if features.ai_background else ""}

**LIFESTYLE treatment** (contextual/environmental):
<div class="frame lifestyle active"><img src="URL" class="lifestyle-img"><div class="lifestyle-overlay"></div><div class="text-area">...</div></div>

{"**CTA FRAME (final frame)** - with animated button:" if features.cta_button else ""}
{'''<div class="frame active">
  <div class="product-wrap"><img src="URL" class="product-img"></div>
  <div class="text-area">
    <h1>Ready to <span class="text-gradient">Elevate</span>?</h1>
    <button class="cta-button">SHOP NOW</button>
  </div>
</div>''' if features.cta_button else ""}

FRAME STRUCTURE:
1. HERO: Impactful opening - lifestyle OR dramatic product reveal
2. FEATURE: Product detail + benefit {"(use accent-line)" if features.text_effects else ""}
3. VALUE: Social proof or key differentiator
4. CTA: {"FINAL frame with animated CTA button - MUST include <button class='cta-button'>" if features.cta_button else "Strong closing frame"}

⚠️ SAFE ZONE RULES (TEXT ONLY - products can fill entire frame):
- PRODUCTS/IMAGES: Can extend to ALL edges, fill entire 1080x1920, NO restrictions
- TEXT ONLY: Must be at 15% from bottom (300px), padded from right (200px)
- Products should be LARGE and fill the frame - no empty/blank space
- Only text needs to avoid Instagram UI areas

COMPOSITION:
- Product images: LARGE, fill the frame, can go edge to edge
- Center product vertically, let it dominate the visual space
- Text positioned at bottom 15% in safe zone
- No blank/empty areas - product fills available space

{"🚨 SMART COPY CONSTRAINTS - CRITICAL:" if features.smart_copy else ""}
{"- ONLY use text/copy that is provided in the EXTRACTED COPY section below" if features.smart_copy else ""}
{"- DO NOT invent features, benefits, or claims not found on the page" if features.smart_copy else ""}
{"- DO NOT hallucinate product specifications or capabilities" if features.smart_copy else ""}
{"- If no price is provided, don't show a price" if features.smart_copy else ""}
{"- Use the EXACT product name as extracted" if features.smart_copy else ""}
{"- CTA button text should match the extracted CTA or use generic 'Shop Now'/'Learn More'" if features.smart_copy else ""}
{"- Headlines can be shortened/reformatted but must derive from actual page content" if features.smart_copy else ""}
{"- You CAN use generic premium phrases like 'Elevate Your Style' for transitions" if features.smart_copy else ""}
{"- Feature claims MUST come from the extracted features list" if features.smart_copy else ""}

Add at end: <script>const timing = [3500, 3500, 3500, 3500, 3500];</script>

Return ONLY complete HTML. No explanations."""

    # Build image list
    images_text = "\n".join([f"{i+1}. {img}" for i, img in enumerate(product_images)]) if product_images else "No images found - use placeholder styling"

    # AI background info
    bg_info = f"\nAI BACKGROUND (use as .ai-bg src): {ai_background_url}" if ai_background_url else ""

    # Color info for user prompt
    color_info = ""
    if brand_colors:
        color_info = f"\n🎨 BRAND COLORS EXTRACTED FROM PRODUCT:\n- Primary: {brand_colors['primary']} ({brand_colors['primary_name']})\n- Secondary: {brand_colors['secondary']} ({brand_colors['secondary_name']})\n- Accent: {brand_colors['accent']} ({brand_colors['accent_name']})\nUse these colors for all glows, gradients, and accents to match the product!"

    # Build user prompt with feature-conditional sections
    user_prompt_parts = [
        f"Product: {page_title}",
        f"URL: {url}",
        "",
        "IMAGES (use these exact URLs - ONE per frame):",
        images_text,
    ]

    # For lifestyle, explicitly map images to frames
    if is_lifestyle_product and product_images:
        user_prompt_parts.extend([
            "",
            "📸 IMAGE TO FRAME MAPPING (MUST FOLLOW):",
        ])
        for i, img in enumerate(product_images[:4]):  # Max 4 frames
            frame_label = "FINAL/CTA" if i == min(3, len(product_images)-1) else f"Frame {i+1}"
            user_prompt_parts.append(f"   {frame_label}: {img}")
        user_prompt_parts.append("")
        user_prompt_parts.append("USE A DIFFERENT IMAGE ON EACH FRAME - cycle through the list above!")

    if bg_info:
        user_prompt_parts.append(bg_info)
    if color_info:
        user_prompt_parts.append(color_info)

    if features.smart_copy and smart_copy_text:
        user_prompt_parts.extend([
            "",
            "📝 EXTRACTED COPY (USE ONLY THIS TEXT - NO HALLUCINATION):",
            smart_copy_text,
        ])

    if prompt:
        user_prompt_parts.extend(["", f"EXTRA INSTRUCTIONS: {prompt}"])

    # Add category-specific instructions
    if is_lifestyle_product:
        user_prompt_parts.extend([
            "",
            "🚨🚨🚨 FASHION/LIFESTYLE MODE - STRICT RULES 🚨🚨🚨",
            "",
            "MANDATORY RULES - FOLLOW EXACTLY:",
            "",
            "1. DIFFERENT IMAGE ON EACH FRAME - Use a different image URL for each frame",
            "2. FULL-BLEED IMAGES - Image fills ENTIRE 1080x1920, edge-to-edge",
            "3. ONLY .lifestyle-img CLASS - No .product-wrap, no .product-img",
            "4. NO BACKGROUNDS - No .bg-glow, no .ai-bg, no extra images behind",
            "5. ONE HEADLINE PER FRAME - Never stack multiple headlines",
            "6. SHOP NOW BUTTON ONLY ON FINAL FRAME - Not on other frames",
            "7. NO TRUST BADGES - Skip entirely for fashion",
            "8. NO ANIMATIONS on images - Keep them static",
            "",
            "FRAME STRUCTURE (copy exactly):",
            '<div class="frame lifestyle active">',
            '  <img src="[UNIQUE_IMAGE_URL]" class="lifestyle-img">',
            '  <div class="lifestyle-overlay"></div>',
            '  <div class="text-area"><h1>One Headline</h1></div>',
            '</div>',
            "",
            "USE THE IMAGE MAPPING ABOVE:",
            "- FRAME 1: Use image 1 from the list + intro headline",
            "- FRAME 2: Use image 2 from the list + feature headline",
            "- FRAME 3: Use image 3 from the list + benefit headline",
            "- FRAME 4 (FINAL): Use image 4 from the list + CTA button",
            "",
            "⚠️ NEVER repeat the same image - each frame MUST have a DIFFERENT image URL",
            "",
            "FORBIDDEN:",
            "- .bg-glow element",
            "- .ai-bg element",
            "- .product-wrap element",
            "- Multiple headlines in one frame",
            "- Same image on multiple frames",
            "- SHOP NOW on non-final frames",
            "- Any background behind the lifestyle image",
        ])
    else:
        user_prompt_parts.extend([
            "",
            f"🎯 CATEGORY: {product_category.upper()}",
        ])

    user_prompt_parts.extend([
        "",
        "CRITICAL RULES:",
    ])

    if is_lifestyle_product:
        user_prompt_parts.extend([
            "1. DIFFERENT image on each frame - cycle through provided images",
            "2. FULL-BLEED using .lifestyle-img - image IS the background",
            "3. ONE headline per frame - no stacking text",
            "4. CTA button ONLY on final frame",
            "5. NO .bg-glow, NO .ai-bg, NO backgrounds behind image",
        ])
    else:
        user_prompt_parts.extend([
            "1. Product images should be LARGE (950px wide, up to 1200px tall) and FILL the frame",
            "2. No blank space - products fill available area",
            "3. Text in safe zone at bottom",
        ])

    rule_num = 4
    if features.smart_copy:
        user_prompt_parts.append(f"{rule_num}. ONLY use text from the EXTRACTED COPY section above - do not invent features or claims")
        rule_num += 1

    if features.cta_button:
        user_prompt_parts.append(f"{rule_num}. Final frame MUST have animated CTA button")
        rule_num += 1

    if ai_background_url and features.ai_background:
        user_prompt_parts.append(f"{rule_num}. Use the AI background image on some frames for premium cinematic look.")

    user_prompt = "\n".join(user_prompt_parts)

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


async def render_multi_format(html_content: str, base_video_id: str, fps: int = 30):
    """Render the same HTML content in multiple formats (9:16, 1:1, 16:9)"""
    formats = ["reel", "square", "landscape"]

    for fmt in formats:
        fmt_video_id = f"{base_video_id}_{fmt}"
        video_jobs[fmt_video_id] = {"status": "queued", "progress": 0, "format": fmt}

    for fmt in formats:
        fmt_video_id = f"{base_video_id}_{fmt}"
        print(f"🎬 [{fmt_video_id}] Starting {fmt} format render...")
        await render_video_from_html(html_content, fmt_video_id, fmt, fps)

    # Update the base job to track all formats
    video_jobs[base_video_id] = {
        "status": "complete",
        "progress": 100,
        "formats": {
            fmt: f"{base_video_id}_{fmt}" for fmt in formats
        }
    }
    print(f"✅ [{base_video_id}] All formats complete!")


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
        body { font-family: 'Inter', -apple-system, sans-serif; background: #0a0a0a; color: white; min-height: 100vh; padding: 40px 20px; }
        .container { max-width: 700px; margin: 0 auto; }
        h1 { font-size: 32px; font-weight: 700; margin-bottom: 8px; }
        .subtitle { color: #888; margin-bottom: 30px; }
        label { display: block; font-size: 14px; color: #888; margin-bottom: 8px; }
        input[type="url"], input[type="text"] { width: 100%; padding: 16px; font-size: 16px; border: 1px solid #333; border-radius: 8px; background: #111; color: white; margin-bottom: 20px; }
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

        /* Feature toggles */
        .features-section { margin: 25px 0; padding: 20px; background: #111; border-radius: 12px; border: 1px solid #222; }
        .features-section h3 { font-size: 16px; margin-bottom: 15px; color: #ccc; display: flex; align-items: center; gap: 8px; }
        .features-section h3::before { content: '⚙️'; }
        .features-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
        .feature-toggle { display: flex; align-items: center; gap: 10px; padding: 10px 12px; background: #1a1a1a; border-radius: 8px; cursor: pointer; transition: background 0.2s; }
        .feature-toggle:hover { background: #222; }
        .feature-toggle input { display: none; }
        .feature-toggle .toggle-switch { width: 40px; height: 22px; background: #333; border-radius: 11px; position: relative; transition: background 0.2s; flex-shrink: 0; }
        .feature-toggle .toggle-switch::after { content: ''; position: absolute; width: 18px; height: 18px; background: #666; border-radius: 50%; top: 2px; left: 2px; transition: transform 0.2s, background 0.2s; }
        .feature-toggle input:checked + .toggle-switch { background: #6366f1; }
        .feature-toggle input:checked + .toggle-switch::after { transform: translateX(18px); background: white; }
        .feature-toggle .label { font-size: 13px; color: #aaa; }
        .feature-toggle .label strong { display: block; color: white; font-size: 14px; margin-bottom: 2px; }
        .feature-toggle.highlight { background: linear-gradient(135deg, rgba(99,102,241,0.1), rgba(236,72,153,0.1)); border: 1px solid rgba(99,102,241,0.3); }
        .feature-toggle.highlight .label strong { background: linear-gradient(135deg, #6366f1, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

        .multi-download { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 15px; }
        .multi-download a { padding: 10px 18px; background: #22c55e; border-radius: 6px; color: white; text-decoration: none; font-weight: 600; font-size: 14px; }
        .multi-download a:hover { background: #16a34a; }
        .multi-download a.reel { background: linear-gradient(135deg, #ec4899, #f43f5e); }
        .multi-download a.square { background: linear-gradient(135deg, #6366f1, #8b5cf6); }
        .multi-download a.landscape { background: linear-gradient(135deg, #14b8a6, #22c55e); }

        .api-info { margin-top: 40px; padding-top: 25px; border-top: 1px solid #222; }
        .api-info h3 { font-size: 16px; margin-bottom: 15px; }
        code { background: #1a1a1a; padding: 2px 6px; border-radius: 4px; font-size: 13px; }

        /* Style controls */
        .style-section { margin: 25px 0; padding: 20px; background: #111; border-radius: 12px; border: 1px solid #222; }
        .style-section h3 { font-size: 16px; margin-bottom: 15px; color: #ccc; display: flex; align-items: center; gap: 8px; }
        .style-section h3::before { content: '🎨'; }
        .style-row { display: flex; gap: 20px; margin-bottom: 15px; }
        .style-control { flex: 1; }
        .style-control label { display: block; font-size: 13px; color: #888; margin-bottom: 6px; }
        .style-control select { width: 100%; padding: 12px; font-size: 14px; border: 1px solid #333; border-radius: 8px; background: #1a1a1a; color: white; cursor: pointer; }
        .style-control select:focus { outline: none; border-color: #6366f1; }
        .style-control input[type="color"] { width: 100%; height: 44px; padding: 4px; border: 1px solid #333; border-radius: 8px; background: #1a1a1a; cursor: pointer; }
        .color-preview { display: flex; align-items: center; gap: 10px; }
        .color-preview span { font-size: 13px; color: #888; }

        @media (max-width: 600px) {
            .features-grid { grid-template-columns: 1fr; }
            .style-row { flex-direction: column; gap: 15px; }
        }
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

            <div class="features-section">
                <h3>Premium Features</h3>
                <div class="features-grid">
                    <label class="feature-toggle">
                        <input type="checkbox" id="feat_bg_removal" checked>
                        <span class="toggle-switch"></span>
                        <span class="label"><strong>Background Removal</strong>Remove product backgrounds</span>
                    </label>
                    <label class="feature-toggle">
                        <input type="checkbox" id="feat_ai_background" checked>
                        <span class="toggle-switch"></span>
                        <span class="label"><strong>AI Background</strong>Cinematic AI backgrounds</span>
                    </label>
                    <label class="feature-toggle">
                        <input type="checkbox" id="feat_color_extraction" checked>
                        <span class="toggle-switch"></span>
                        <span class="label"><strong>Color Extraction</strong>Match brand colors</span>
                    </label>
                    <label class="feature-toggle">
                        <input type="checkbox" id="feat_cta_button" checked>
                        <span class="toggle-switch"></span>
                        <span class="label"><strong>CTA Button</strong>Animated call-to-action</span>
                    </label>
                    <label class="feature-toggle">
                        <input type="checkbox" id="feat_progress_bar" checked>
                        <span class="toggle-switch"></span>
                        <span class="label"><strong>Progress Bar</strong>Story-style indicator</span>
                    </label>
                    <label class="feature-toggle">
                        <input type="checkbox" id="feat_text_effects" checked>
                        <span class="toggle-switch"></span>
                        <span class="label"><strong>Text Effects</strong>Gradients & accent lines</span>
                    </label>
                    <label class="feature-toggle">
                        <input type="checkbox" id="feat_floating_animation" checked>
                        <span class="toggle-switch"></span>
                        <span class="label"><strong>Float Animation</strong>Product floating effect</span>
                    </label>
                    <label class="feature-toggle">
                        <input type="checkbox" id="feat_smart_copy" checked>
                        <span class="toggle-switch"></span>
                        <span class="label"><strong>Smart Copy</strong>Use only real page text</span>
                    </label>
                    <label class="feature-toggle">
                        <input type="checkbox" id="feat_price_badge" checked>
                        <span class="toggle-switch"></span>
                        <span class="label"><strong>Price Badge</strong>Animated price display</span>
                    </label>
                    <label class="feature-toggle">
                        <input type="checkbox" id="feat_trust_badges" checked>
                        <span class="toggle-switch"></span>
                        <span class="label"><strong>Trust Badges</strong>Free shipping, authentic</span>
                    </label>
                    <label class="feature-toggle highlight">
                        <input type="checkbox" id="feat_multi_format">
                        <span class="toggle-switch"></span>
                        <span class="label"><strong>Multi-Format</strong>9:16 + 1:1 + 16:9</span>
                    </label>
                </div>
            </div>

            <div class="style-section">
                <h3>Text Styling</h3>
                <div class="style-row">
                    <div class="style-control">
                        <label>Font Family</label>
                        <select id="fontSelect">
                            <option value="Inter">Inter (Modern)</option>
                            <option value="Playfair Display">Playfair Display (Elegant)</option>
                            <option value="Montserrat">Montserrat (Clean)</option>
                            <option value="Oswald">Oswald (Bold)</option>
                            <option value="Roboto">Roboto (Neutral)</option>
                            <option value="Poppins">Poppins (Friendly)</option>
                            <option value="Bebas Neue">Bebas Neue (Impact)</option>
                            <option value="Cormorant Garamond">Cormorant Garamond (Luxury)</option>
                        </select>
                    </div>
                    <div class="style-control">
                        <label>Text Color</label>
                        <input type="color" id="textColor" value="#ffffff">
                    </div>
                    <div class="style-control">
                        <label>Accent Color</label>
                        <input type="color" id="accentColor" value="#c9a96e">
                    </div>
                </div>
            </div>

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

        // Collect feature toggles and style options
        function getFeatures() {
            return {
                background_removal: document.getElementById('feat_bg_removal').checked,
                ai_background: document.getElementById('feat_ai_background').checked,
                color_extraction: document.getElementById('feat_color_extraction').checked,
                cta_button: document.getElementById('feat_cta_button').checked,
                progress_bar: document.getElementById('feat_progress_bar').checked,
                text_effects: document.getElementById('feat_text_effects').checked,
                floating_animation: document.getElementById('feat_floating_animation').checked,
                ken_burns: true, // Always on for now
                smart_copy: document.getElementById('feat_smart_copy').checked,
                price_badge: document.getElementById('feat_price_badge').checked,
                trust_badges: document.getElementById('feat_trust_badges').checked,
                multi_format: document.getElementById('feat_multi_format').checked,
                // Style options
                font_family: document.getElementById('fontSelect').value,
                text_color: document.getElementById('textColor').value,
                accent_color: document.getElementById('accentColor').value
            };
        }

        let currentMultiFormat = false;
        let currentFormatIds = null;

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const url = document.getElementById('urlInput').value;
            const prompt = document.getElementById('promptInput').value;
            const features = getFeatures();

            submitBtn.disabled = true;
            submitBtn.textContent = features.multi_format ? 'Generating 3 Formats...' : 'Generating...';
            statusDiv.classList.remove('hidden');
            statusText.textContent = features.multi_format ? 'Starting multi-format generation...' : 'Starting video generation...';
            progressFill.style.width = '0%';
            downloadArea.innerHTML = '';

            try {
                const res = await fetch('/generate-from-url', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url, prompt, features })
                });
                const data = await res.json();

                if (data.video_id) {
                    currentMultiFormat = data.multi_format || false;
                    currentFormatIds = data.format_ids || null;
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
                // For multi-format, poll the landscape format (last to complete)
                const pollId = currentMultiFormat ? currentFormatIds.landscape : videoId;
                const res = await fetch('/status/' + pollId);
                const data = await res.json();

                progressFill.style.width = data.progress + '%';

                if (data.status === 'complete') {
                    progressFill.style.width = '100%';

                    if (currentMultiFormat && currentFormatIds) {
                        statusText.textContent = 'All 3 formats ready!';
                        downloadArea.innerHTML = `
                            <div class="multi-download">
                                <a href="/download/${currentFormatIds.reel}" class="reel" download>📱 Reel (9:16)</a>
                                <a href="/download/${currentFormatIds.square}" class="square" download>⬜ Square (1:1)</a>
                                <a href="/download/${currentFormatIds.landscape}" class="landscape" download>🖥️ Landscape (16:9)</a>
                            </div>
                        `;
                    } else {
                        statusText.textContent = 'Video ready!';
                        downloadArea.innerHTML = '<a href="/download/' + videoId + '" class="download-btn" download>Download Video</a>';
                    }

                    submitBtn.disabled = false;
                    submitBtn.textContent = 'Generate Another';
                } else if (data.status === 'error') {
                    statusText.innerHTML = '<span class="error">Error: ' + data.error + '</span>';
                    submitBtn.disabled = false;
                    submitBtn.textContent = 'Try Again';
                } else {
                    const formatLabel = currentMultiFormat ? ' (3 formats)' : '';
                    statusText.textContent = 'Generating' + formatLabel + '... ' + data.status + ' (' + data.progress + '%)';
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

        # Pass features to the HTML generator
        html_content = await generate_html_from_url(request.url, request.prompt, request.features)
        print(f"✅ [{video_id}] HTML generated ({len(html_content)} chars)")

        # Check if multi-format export is enabled
        if request.features.multi_format:
            print(f"🎬 [{video_id}] Multi-format mode enabled - generating 3 formats")
            background_tasks.add_task(render_multi_format, html_content, video_id, request.fps)
            return {
                "message": "Multi-format video generation started",
                "status": "processing",
                "video_id": video_id,
                "record_id": request.record_id,
                "url": request.url,
                "multi_format": True,
                "formats": ["reel", "square", "landscape"],
                "format_ids": {
                    "reel": f"{video_id}_reel",
                    "square": f"{video_id}_square",
                    "landscape": f"{video_id}_landscape"
                },
                "features": request.features.model_dump()
            }
        else:
            background_tasks.add_task(render_video_from_html, html_content, video_id, request.format, request.fps)
            return {
                "message": "Video generation started",
                "status": "processing",
                "video_id": video_id,
                "record_id": request.record_id,
                "url": request.url,
                "format": request.format,
                "features": request.features.model_dump()
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
