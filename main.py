import os
import uuid
import asyncio
import subprocess
import json
import imageio_ffmpeg
import httpx
import base64
import io
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
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
import random

# --- RETRY HELPER WITH EXPONENTIAL BACKOFF ---
async def retry_with_backoff(func, max_retries: int = 4, base_delay: float = 2.0):
    """
    Retry an async function with exponential backoff.
    Delays: 2s, 4s, 8s, 16s (with jitter)
    """
    last_exception = None
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                # Exponential backoff with jitter
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"⚠️ Attempt {attempt + 1} failed, retrying in {delay:.1f}s: {e}")
                await asyncio.sleep(delay)
            else:
                print(f"❌ All {max_retries} attempts failed: {e}")
    raise last_exception

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

# --- TEMPLATE COMPONENT SYSTEM ---
# A robust system for building video templates from reusable components


class ComponentCategory(str, Enum):
    """Categories for organizing components"""
    FRAME = "frame"           # Full frame layouts
    ELEMENT = "element"       # UI elements (badges, buttons)
    TEXT = "text"             # Text/headline variations
    OVERLAY = "overlay"       # Visual overlays
    CONTAINER = "container"   # Structural containers


@dataclass
class ComponentProp:
    """Definition of a component property"""
    name: str
    description: str
    required: bool = False
    default: Any = None
    prop_type: str = "string"  # string, url, number, boolean, html


@dataclass
class Component:
    """A reusable template component with metadata and variants"""
    name: str
    description: str
    template: str
    category: ComponentCategory
    props: List[ComponentProp] = field(default_factory=list)
    variants: Dict[str, str] = field(default_factory=dict)
    slots: List[str] = field(default_factory=list)  # Named slots for nested components
    tags: List[str] = field(default_factory=list)   # For filtering/searching

    def render(self, variant: str = None, **kwargs) -> str:
        """Render the component with provided props"""
        # Use variant template if specified
        tmpl = self.variants.get(variant, self.template) if variant else self.template

        # Apply defaults for missing props
        for prop in self.props:
            if prop.name not in kwargs and prop.default is not None:
                kwargs[prop.name] = prop.default

        # Validate required props
        missing = [p.name for p in self.props if p.required and p.name not in kwargs]
        if missing:
            raise ValueError(f"Component '{self.name}' missing required props: {missing}")

        # Replace placeholders
        result = tmpl
        for key, value in kwargs.items():
            result = result.replace(f"{{{key}}}", str(value) if value is not None else "")

        return result

    def get_prop_names(self) -> List[str]:
        """Get list of all prop names"""
        return [p.name for p in self.props]

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses"""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "props": [
                {
                    "name": p.name,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default,
                    "type": p.prop_type
                }
                for p in self.props
            ],
            "variants": list(self.variants.keys()),
            "slots": self.slots,
            "tags": self.tags,
            "template": self.template
        }


class ComponentRegistry:
    """Registry for managing and accessing components"""

    def __init__(self):
        self._components: Dict[str, Component] = {}

    def register(self, component: Component) -> None:
        """Register a component"""
        self._components[component.name] = component

    def get(self, name: str) -> Component:
        """Get a component by name"""
        if name not in self._components:
            raise ValueError(f"Component '{name}' not found. Available: {list(self._components.keys())}")
        return self._components[name]

    def list(self, category: ComponentCategory = None, tags: List[str] = None) -> List[Component]:
        """List components, optionally filtered by category or tags"""
        components = list(self._components.values())

        if category:
            components = [c for c in components if c.category == category]

        if tags:
            components = [c for c in components if any(t in c.tags for t in tags)]

        return components

    def render(self, name: str, variant: str = None, **kwargs) -> str:
        """Render a component by name"""
        return self.get(name).render(variant=variant, **kwargs)

    def list_names(self) -> List[str]:
        """Get all component names"""
        return list(self._components.keys())

    def to_dict(self) -> dict:
        """Convert registry to dictionary for API"""
        return {
            name: comp.to_dict()
            for name, comp in self._components.items()
        }


# Global component registry
component_registry = ComponentRegistry()


# --- REGISTER FRAME COMPONENTS ---

component_registry.register(Component(
    name="product_hero",
    description="Hero product frame with floating animation and brand glow",
    category=ComponentCategory.FRAME,
    tags=["hero", "product", "animated"],
    props=[
        ComponentProp("image_url", "Product image URL", required=True, prop_type="url"),
        ComponentProp("product_name", "Product name for alt text", default="Product"),
        ComponentProp("headline", "Main headline text", required=True),
        ComponentProp("subheadline", "Secondary text", default=""),
    ],
    slots=["badges"],  # Can insert badges component
    template='''<div class="frame active">
  <div class="bg-glow"></div>
  <div class="product-wrap">
    <img src="{image_url}" class="product-img" alt="{product_name}">
  </div>
  <div class="text-area">
    <h1 class="text-clamp-2">{headline}</h1>
    <p class="text-clamp-3">{subheadline}</p>
    <div class="accent-line"></div>
  </div>
  {badges}
</div>''',
    variants={
        "minimal": '''<div class="frame active">
  <div class="bg-glow"></div>
  <div class="product-wrap">
    <img src="{image_url}" class="product-img" alt="{product_name}">
  </div>
  <div class="text-area">
    <h1 class="text-clamp-2">{headline}</h1>
  </div>
</div>''',
        "with_price": '''<div class="frame active">
  <div class="bg-glow"></div>
  <div class="product-wrap">
    <img src="{image_url}" class="product-img" alt="{product_name}">
  </div>
  <div class="text-area">
    <h1 class="text-clamp-2">{headline}</h1>
    <p class="text-clamp-3">{subheadline}</p>
    <div class="accent-line"></div>
  </div>
  <div class="price-badge"><span class="price-current">{price}</span></div>
</div>'''
    }
))

component_registry.register(Component(
    name="lifestyle",
    description="Full-bleed lifestyle image with gradient overlay",
    category=ComponentCategory.FRAME,
    tags=["lifestyle", "fullbleed", "fashion"],
    props=[
        ComponentProp("image_url", "Lifestyle image URL", required=True, prop_type="url"),
        ComponentProp("product_name", "Product name for alt text", default="Product"),
        ComponentProp("headline", "Main headline text", required=True),
        ComponentProp("subheadline", "Secondary text", default=""),
    ],
    template='''<div class="frame lifestyle active">
  <img src="{image_url}" class="lifestyle-img" alt="{product_name}">
  <div class="lifestyle-overlay"></div>
  <div class="text-area">
    <h1 class="text-clamp-2">{headline}</h1>
    <p class="text-clamp-3">{subheadline}</p>
  </div>
</div>''',
    variants={
        "dark_overlay": '''<div class="frame lifestyle active">
  <img src="{image_url}" class="lifestyle-img" alt="{product_name}">
  <div class="lifestyle-overlay" style="background: linear-gradient(to top, rgba(0,0,0,0.9) 0%, rgba(0,0,0,0.5) 50%, transparent 100%);"></div>
  <div class="text-area">
    <h1 class="text-clamp-2">{headline}</h1>
    <p class="text-clamp-3">{subheadline}</p>
  </div>
</div>''',
        "no_text": '''<div class="frame lifestyle active">
  <img src="{image_url}" class="lifestyle-img" alt="{product_name}">
  <div class="lifestyle-overlay"></div>
</div>'''
    }
))

component_registry.register(Component(
    name="ai_background",
    description="Product with AI-generated cinematic background",
    category=ComponentCategory.FRAME,
    tags=["ai", "cinematic", "premium"],
    props=[
        ComponentProp("image_url", "Product image URL", required=True, prop_type="url"),
        ComponentProp("ai_bg_url", "AI background URL", required=True, prop_type="url"),
        ComponentProp("product_name", "Product name for alt text", default="Product"),
        ComponentProp("headline", "Main headline text", required=True),
        ComponentProp("subheadline", "Secondary text", default=""),
    ],
    template='''<div class="frame ai-background active">
  <img src="{ai_bg_url}" class="ai-bg" alt="background">
  <div class="ai-bg-overlay"></div>
  <div class="product-wrap">
    <img src="{image_url}" class="product-img" alt="{product_name}">
  </div>
  <div class="text-area">
    <h1 class="text-clamp-2">{headline}</h1>
    <p class="text-clamp-3">{subheadline}</p>
  </div>
</div>'''
))

component_registry.register(Component(
    name="cta_frame",
    description="Call-to-action frame with animated button",
    category=ComponentCategory.FRAME,
    tags=["cta", "final", "action"],
    props=[
        ComponentProp("image_url", "Product image URL", required=True, prop_type="url"),
        ComponentProp("product_name", "Product name for alt text", default="Product"),
        ComponentProp("headline", "Main headline text", required=True),
        ComponentProp("cta_text", "Button text", default="Shop Now"),
    ],
    template='''<div class="frame active">
  <div class="bg-glow"></div>
  <div class="product-wrap">
    <img src="{image_url}" class="product-img" alt="{product_name}">
  </div>
  <div class="text-area">
    <h1 class="text-clamp-2">{headline}</h1>
    <button class="cta-button">{cta_text}</button>
  </div>
</div>''',
    variants={
        "urgent": '''<div class="frame active">
  <div class="bg-glow"></div>
  <div class="product-wrap">
    <img src="{image_url}" class="product-img" alt="{product_name}">
  </div>
  <div class="text-area">
    <h1 class="text-clamp-2">{headline}</h1>
    <p class="text-accent" style="font-size: 24px; margin-bottom: 20px;">Limited Time Only</p>
    <button class="cta-button">{cta_text}</button>
  </div>
</div>''',
        "with_trust": '''<div class="frame active">
  <div class="bg-glow"></div>
  <div class="product-wrap">
    <img src="{image_url}" class="product-img" alt="{product_name}">
  </div>
  <div class="text-area">
    <h1 class="text-clamp-2">{headline}</h1>
    <button class="cta-button">{cta_text}</button>
  </div>
  <div class="trust-badges">
    <div class="trust-badge"><span class="icon">🚚</span><span class="text">Free Shipping</span></div>
    <div class="trust-badge"><span class="icon">✓</span><span class="text">Authentic</span></div>
  </div>
</div>'''
    }
))

component_registry.register(Component(
    name="feature_frame",
    description="Feature highlight with accent styling",
    category=ComponentCategory.FRAME,
    tags=["feature", "benefit", "detail"],
    props=[
        ComponentProp("image_url", "Product image URL", required=True, prop_type="url"),
        ComponentProp("product_name", "Product name for alt text", default="Product"),
        ComponentProp("headline", "Main headline text", required=True),
        ComponentProp("feature_text", "Feature description", default=""),
    ],
    template='''<div class="frame active">
  <div class="bg-glow"></div>
  <div class="product-wrap">
    <img src="{image_url}" class="product-img" alt="{product_name}">
  </div>
  <div class="text-area">
    <h1 class="text-gradient text-clamp-2">{headline}</h1>
    <p class="subtitle-brand text-clamp-3">{feature_text}</p>
    <div class="accent-line"></div>
  </div>
</div>'''
))

component_registry.register(Component(
    name="split_frame",
    description="Split layout with image on one side, text on other",
    category=ComponentCategory.FRAME,
    tags=["split", "editorial", "modern"],
    props=[
        ComponentProp("image_url", "Product image URL", required=True, prop_type="url"),
        ComponentProp("product_name", "Product name for alt text", default="Product"),
        ComponentProp("headline", "Main headline text", required=True),
        ComponentProp("body_text", "Body copy", default=""),
    ],
    template='''<div class="frame active" style="display: grid; grid-template-columns: 1fr 1fr; gap: 0;">
  <div style="display: flex; align-items: center; justify-content: center; padding: 60px;">
    <img src="{image_url}" style="max-width: 100%; max-height: 80%; object-fit: contain;" alt="{product_name}">
  </div>
  <div style="display: flex; flex-direction: column; justify-content: center; padding: 60px; background: rgba(0,0,0,0.3);">
    <h1 class="text-clamp-3" style="font-size: 48px; margin-bottom: 24px;">{headline}</h1>
    <p class="text-clamp-4" style="font-size: 24px; opacity: 0.8;">{body_text}</p>
  </div>
</div>'''
))


# --- REGISTER ELEMENT COMPONENTS ---

component_registry.register(Component(
    name="price_badge",
    description="Price display with optional discount",
    category=ComponentCategory.ELEMENT,
    tags=["price", "badge", "discount"],
    props=[
        ComponentProp("current_price", "Current price", required=True),
        ComponentProp("original_price", "Original price (for sale)", default=""),
        ComponentProp("discount_percent", "Discount percentage", default=""),
    ],
    template='''<div class="price-badge">
  <span class="price-original">{original_price}</span>
  <span class="price-current">{current_price}</span>
  <span class="price-discount">{discount_percent}% OFF</span>
</div>''',
    variants={
        "simple": '''<div class="price-badge">
  <span class="price-current">{current_price}</span>
</div>''',
        "large": '''<div class="price-badge" style="transform: scale(1.3);">
  <span class="price-current">{current_price}</span>
</div>'''
    }
))

component_registry.register(Component(
    name="trust_badges",
    description="Trust indicators for credibility",
    category=ComponentCategory.ELEMENT,
    tags=["trust", "badges", "credibility"],
    props=[
        ComponentProp("badge1_icon", "First badge icon", default="🚚"),
        ComponentProp("badge1_text", "First badge text", default="Free Shipping"),
        ComponentProp("badge2_icon", "Second badge icon", default="✓"),
        ComponentProp("badge2_text", "Second badge text", default="100% Authentic"),
        ComponentProp("badge3_icon", "Third badge icon", default="⚡"),
        ComponentProp("badge3_text", "Third badge text", default="Fast Delivery"),
    ],
    template='''<div class="trust-badges">
  <div class="trust-badge"><span class="icon">{badge1_icon}</span><span class="text">{badge1_text}</span></div>
  <div class="trust-badge"><span class="icon">{badge2_icon}</span><span class="text">{badge2_text}</span></div>
  <div class="trust-badge"><span class="icon">{badge3_icon}</span><span class="text">{badge3_text}</span></div>
</div>''',
    variants={
        "two_badges": '''<div class="trust-badges">
  <div class="trust-badge"><span class="icon">{badge1_icon}</span><span class="text">{badge1_text}</span></div>
  <div class="trust-badge"><span class="icon">{badge2_icon}</span><span class="text">{badge2_text}</span></div>
</div>''',
        "compact": '''<div class="trust-badges" style="font-size: 14px; gap: 10px;">
  <div class="trust-badge"><span class="icon">{badge1_icon}</span><span class="text">{badge1_text}</span></div>
  <div class="trust-badge"><span class="icon">{badge2_icon}</span><span class="text">{badge2_text}</span></div>
</div>'''
    }
))

component_registry.register(Component(
    name="cta_button",
    description="Animated call-to-action button",
    category=ComponentCategory.ELEMENT,
    tags=["button", "cta", "action"],
    props=[
        ComponentProp("text", "Button text", default="Shop Now"),
        ComponentProp("style", "Additional inline styles", default=""),
    ],
    template='''<button class="cta-button" style="{style}">{text}</button>''',
    variants={
        "outline": '''<button class="cta-button" style="background: transparent; border: 2px solid currentColor; {style}">{text}</button>''',
        "large": '''<button class="cta-button" style="padding: 28px 70px; font-size: 32px; {style}">{text}</button>''',
        "pill": '''<button class="cta-button" style="border-radius: 100px; {style}">{text}</button>'''
    }
))

component_registry.register(Component(
    name="progress_bar",
    description="Video progress indicator",
    category=ComponentCategory.ELEMENT,
    tags=["progress", "navigation", "indicator"],
    props=[
        ComponentProp("total_segments", "Number of segments", default="4", prop_type="number"),
        ComponentProp("active_segment", "Currently active segment (1-based)", default="1", prop_type="number"),
    ],
    template='''<div class="progress-bar">
  <div class="progress-segment active"></div>
  <div class="progress-segment"></div>
  <div class="progress-segment"></div>
  <div class="progress-segment"></div>
</div>'''
))


# --- REGISTER TEXT COMPONENTS ---

component_registry.register(Component(
    name="headline",
    description="Styled headline with various treatments",
    category=ComponentCategory.TEXT,
    tags=["headline", "text", "title"],
    props=[
        ComponentProp("text", "Headline text", required=True),
        ComponentProp("size", "Size class (headline-sm, headline-lg)", default=""),
    ],
    template='''<h1 class="text-clamp-2 {size}">{text}</h1>''',
    variants={
        "gradient": '''<h1 class="text-gradient text-clamp-2 {size}">{text}</h1>''',
        "gradient_bold": '''<h1 class="text-gradient-bold text-clamp-2 {size}">{text}</h1>''',
        "brand": '''<h1 class="text-brand text-clamp-2 {size}">{text}</h1>''',
        "accent": '''<h1 class="text-accent text-clamp-2 {size}">{text}</h1>'''
    }
))

component_registry.register(Component(
    name="headline_highlight",
    description="Headline with a highlighted keyword",
    category=ComponentCategory.TEXT,
    tags=["headline", "highlight", "emphasis"],
    props=[
        ComponentProp("prefix", "Text before highlight", default=""),
        ComponentProp("keyword", "Highlighted word", required=True),
        ComponentProp("suffix", "Text after highlight", default=""),
    ],
    template='''<h1 class="text-clamp-2">{prefix} <span class="highlight">{keyword}</span> {suffix}</h1>'''
))

component_registry.register(Component(
    name="subtitle",
    description="Secondary text/subtitle",
    category=ComponentCategory.TEXT,
    tags=["subtitle", "text", "secondary"],
    props=[
        ComponentProp("text", "Subtitle text", required=True),
    ],
    template='''<p class="text-clamp-3">{text}</p>''',
    variants={
        "brand": '''<p class="subtitle-brand text-clamp-3">{text}</p>''',
        "small": '''<p class="text-clamp-3" style="font-size: 24px;">{text}</p>''',
        "large": '''<p class="text-clamp-3" style="font-size: 40px;">{text}</p>'''
    }
))


# --- REGISTER OVERLAY COMPONENTS ---

component_registry.register(Component(
    name="cinematic_overlays",
    description="Full cinematic overlay package (vignette, grain, color grade)",
    category=ComponentCategory.OVERLAY,
    tags=["overlay", "cinematic", "premium"],
    props=[],
    template='''<div class="vignette"></div>
<div class="film-grain"></div>
<div class="color-grade"></div>'''
))

component_registry.register(Component(
    name="vignette",
    description="Edge darkening vignette effect",
    category=ComponentCategory.OVERLAY,
    tags=["overlay", "vignette"],
    props=[
        ComponentProp("intensity", "Vignette darkness (0-1)", default="0.25", prop_type="number"),
    ],
    template='''<div class="vignette" style="background: radial-gradient(ellipse at center, transparent 60%, rgba(0,0,0,{intensity}) 100%);"></div>'''
))

component_registry.register(Component(
    name="gradient_overlay",
    description="Customizable gradient overlay",
    category=ComponentCategory.OVERLAY,
    tags=["overlay", "gradient"],
    props=[
        ComponentProp("direction", "Gradient direction (to top, to bottom, etc.)", default="to top"),
        ComponentProp("color", "Overlay color", default="0,0,0"),
        ComponentProp("opacity_start", "Start opacity", default="0.8"),
        ComponentProp("opacity_end", "End opacity", default="0"),
    ],
    template='''<div style="position: absolute; bottom: 0; left: 0; width: 100%; height: 50%; background: linear-gradient({direction}, rgba({color},{opacity_start}) 0%, rgba({color},{opacity_end}) 100%); z-index: 1; pointer-events: none;"></div>'''
))


# --- REGISTER CONTAINER COMPONENTS ---

component_registry.register(Component(
    name="reel_container",
    description="Base container for Instagram Reel videos",
    category=ComponentCategory.CONTAINER,
    tags=["container", "reel", "base"],
    props=[
        ComponentProp("frames", "HTML content for frames", required=True, prop_type="html"),
        ComponentProp("include_overlays", "Include cinematic overlays", default="true", prop_type="boolean"),
    ],
    template='''<div class="reel-container">
  <div class="bg-glow"></div>
  {frames}
  <div class="vignette"></div>
  <div class="film-grain"></div>
  <div class="color-grade"></div>
</div>''',
    variants={
        "minimal": '''<div class="reel-container">
  {frames}
</div>''',
        "no_glow": '''<div class="reel-container">
  {frames}
  <div class="vignette"></div>
  <div class="film-grain"></div>
  <div class="color-grade"></div>
</div>'''
    }
))

component_registry.register(Component(
    name="text_area",
    description="Safe text area container",
    category=ComponentCategory.CONTAINER,
    tags=["container", "text", "safe-zone"],
    props=[
        ComponentProp("content", "Text content HTML", required=True, prop_type="html"),
        ComponentProp("position", "Position (bottom, center, top)", default="bottom"),
    ],
    template='''<div class="text-area">
  {content}
</div>''',
    variants={
        "center": '''<div class="text-area" style="bottom: auto; top: 50%; transform: translateY(-50%);">
  {content}
</div>''',
        "top": '''<div class="text-area" style="bottom: auto; top: 280px;">
  {content}
</div>'''
    }
))


# --- VIDEO PRESETS ---
# Pre-configured video structures for common use cases

VIDEO_PRESETS = {
    "product_showcase": {
        "name": "Product Showcase",
        "description": "4-frame product video: Hero → Feature → Benefit → CTA",
        "frames": ["product_hero", "feature_frame", "feature_frame", "cta_frame"],
        "timing": [5500, 4000, 4000, 4500],
        "best_for": ["products", "e-commerce", "launches"]
    },
    "lifestyle_story": {
        "name": "Lifestyle Story",
        "description": "4-frame lifestyle video: Lifestyle → Product → Feature → CTA",
        "frames": ["lifestyle", "product_hero", "feature_frame", "cta_frame"],
        "timing": [5000, 4500, 4000, 4500],
        "best_for": ["fashion", "beauty", "home"]
    },
    "quick_promo": {
        "name": "Quick Promo",
        "description": "3-frame quick promo: Hero → Feature → CTA",
        "frames": ["product_hero", "feature_frame", "cta_frame"],
        "timing": [4000, 3500, 3500],
        "best_for": ["sales", "promotions", "flash deals"]
    },
    "premium_cinematic": {
        "name": "Premium Cinematic",
        "description": "4-frame cinematic video with AI backgrounds",
        "frames": ["ai_background", "feature_frame", "lifestyle", "cta_frame"],
        "timing": [6000, 4500, 4500, 5000],
        "best_for": ["luxury", "premium", "high-end"]
    },
    "social_teaser": {
        "name": "Social Teaser",
        "description": "2-frame quick teaser: Impact → CTA",
        "frames": ["product_hero", "cta_frame"],
        "timing": [4000, 3000],
        "best_for": ["teasers", "stories", "quick ads"]
    }
}


class VideoBuilder:
    """Fluent builder for assembling videos from components"""

    def __init__(self):
        self.frames: List[str] = []
        self.timing: List[int] = []
        self.global_props: Dict[str, Any] = {}

    def set_props(self, **props) -> 'VideoBuilder':
        """Set global props that apply to all frames"""
        self.global_props.update(props)
        return self

    def add_frame(self, component_name: str, variant: str = None,
                  duration_ms: int = 4000, **props) -> 'VideoBuilder':
        """Add a frame using a component"""
        merged_props = {**self.global_props, **props}
        frame_html = component_registry.render(component_name, variant=variant, **merged_props)
        self.frames.append(frame_html)
        self.timing.append(duration_ms)
        return self

    def from_preset(self, preset_name: str, images: List[str],
                    headlines: List[str], **props) -> 'VideoBuilder':
        """Build video from a preset template"""
        if preset_name not in VIDEO_PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(VIDEO_PRESETS.keys())}")

        preset = VIDEO_PRESETS[preset_name]
        self.global_props.update(props)

        for i, (frame_name, duration) in enumerate(zip(preset["frames"], preset["timing"])):
            frame_props = {
                "image_url": images[i % len(images)] if images else "",
                "headline": headlines[i % len(headlines)] if headlines else "",
                **self.global_props
            }
            self.add_frame(frame_name, duration_ms=duration, **frame_props)

        return self

    def build(self) -> dict:
        """Build the final video structure"""
        frames_html = "\n".join(self.frames)
        container = component_registry.render("reel_container", frames=frames_html)

        return {
            "html": container,
            "timing": self.timing,
            "frame_count": len(self.frames)
        }

    def build_html(self) -> str:
        """Build and return just the HTML"""
        return self.build()["html"]


def get_video_builder() -> VideoBuilder:
    """Get a new VideoBuilder instance"""
    return VideoBuilder()


# --- LEGACY SUPPORT ---
# Keep TEMPLATE_SNIPPETS for backwards compatibility

TEMPLATE_SNIPPETS = {
    # Hero product frame with floating animation and brand glow
    "product_hero": '''<div class="frame active">
  <div class="bg-glow"></div>
  <div class="product-wrap">
    <img src="{image_url}" class="product-img" alt="{product_name}">
  </div>
  <div class="text-area">
    <h1 class="text-clamp-2">{headline}</h1>
    <p class="text-clamp-3">{subheadline}</p>
    <div class="accent-line"></div>
  </div>
</div>''',

    # Lifestyle frame with full-bleed image and gradient overlay
    "lifestyle": '''<div class="frame lifestyle active">
  <img src="{image_url}" class="lifestyle-img" alt="{product_name}">
  <div class="lifestyle-overlay"></div>
  <div class="text-area">
    <h1 class="text-clamp-2">{headline}</h1>
    <p class="text-clamp-3">{subheadline}</p>
  </div>
</div>''',

    # Product with AI-generated cinematic background
    "ai_background": '''<div class="frame ai-background active">
  <img src="{ai_bg_url}" class="ai-bg" alt="background">
  <div class="ai-bg-overlay"></div>
  <div class="product-wrap">
    <img src="{image_url}" class="product-img" alt="{product_name}">
  </div>
  <div class="text-area">
    <h1 class="text-clamp-2">{headline}</h1>
    <p class="text-clamp-3">{subheadline}</p>
  </div>
</div>''',

    # Call-to-action frame with animated button
    "cta_frame": '''<div class="frame active">
  <div class="bg-glow"></div>
  <div class="product-wrap">
    <img src="{image_url}" class="product-img" alt="{product_name}">
  </div>
  <div class="text-area">
    <h1 class="text-clamp-2">{headline}</h1>
    <button class="cta-button">{cta_text}</button>
  </div>
</div>''',

    # Feature highlight frame with accent styling
    "feature_frame": '''<div class="frame active">
  <div class="bg-glow"></div>
  <div class="product-wrap">
    <img src="{image_url}" class="product-img" alt="{product_name}">
  </div>
  <div class="text-area">
    <h1 class="text-gradient text-clamp-2">{headline}</h1>
    <p class="subtitle-brand text-clamp-3">{feature_text}</p>
    <div class="accent-line"></div>
  </div>
</div>''',

    # Price badge component
    "price_badge": '''<div class="price-badge">
  <span class="price-original">{original_price}</span>
  <span class="price-current">{current_price}</span>
  <span class="price-discount">{discount_percent}% OFF</span>
</div>''',

    # Simple price badge (no discount)
    "price_badge_simple": '''<div class="price-badge">
  <span class="price-current">{price}</span>
</div>''',

    # Trust badges component
    "trust_badges": '''<div class="trust-badges">
  <div class="trust-badge"><span class="icon">🚚</span><span class="text">Free Shipping</span></div>
  <div class="trust-badge"><span class="icon">✓</span><span class="text">100% Authentic</span></div>
  <div class="trust-badge"><span class="icon">⚡</span><span class="text">Fast Delivery</span></div>
</div>''',

    # Custom trust badges (configurable)
    "trust_badges_custom": '''<div class="trust-badges">
  {badges}
</div>''',

    # Single trust badge
    "trust_badge_item": '''<div class="trust-badge"><span class="icon">{icon}</span><span class="text">{text}</span></div>''',

    # Progress bar component
    "progress_bar": '''<div class="progress-bar">
  <div class="progress-segment active"></div>
  <div class="progress-segment"></div>
  <div class="progress-segment"></div>
  <div class="progress-segment"></div>
</div>''',

    # Cinematic overlays (add as last children of reel-container)
    "cinematic_overlays": '''<div class="vignette"></div>
<div class="film-grain"></div>
<div class="color-grade"></div>''',

    # Base reel container structure
    "reel_container": '''<div class="reel-container">
  <div class="bg-glow"></div>
  {frames}
  <div class="vignette"></div>
  <div class="film-grain"></div>
  <div class="color-grade"></div>
</div>''',

    # Gradient headline variations
    "headline_gradient": '''<h1 class="text-gradient text-clamp-2">{headline}</h1>''',
    "headline_gradient_bold": '''<h1 class="text-gradient-bold text-clamp-2">{headline}</h1>''',
    "headline_brand": '''<h1 class="text-brand text-clamp-2">{headline}</h1>''',

    # Text with highlighted keyword
    "headline_highlight": '''<h1 class="text-clamp-2">{prefix} <span class="highlight">{keyword}</span> {suffix}</h1>''',
}


def get_template_snippet(name: str, **kwargs) -> str:
    """Get a template snippet with variables replaced.

    Args:
        name: Name of the snippet (e.g., 'product_hero', 'lifestyle', 'cta_frame')
        **kwargs: Variables to replace in the template

    Returns:
        Formatted HTML string with variables replaced

    Example:
        >>> get_template_snippet('product_hero',
        ...     image_url='https://example.com/shoe.png',
        ...     product_name='Nike Air Max',
        ...     headline='Step Into Style',
        ...     subheadline='Comfort meets design')
    """
    if name not in TEMPLATE_SNIPPETS:
        raise ValueError(f"Unknown template snippet: {name}. Available: {list(TEMPLATE_SNIPPETS.keys())}")

    template = TEMPLATE_SNIPPETS[name]

    # Replace all provided variables
    for key, value in kwargs.items():
        template = template.replace(f"{{{key}}}", str(value) if value else "")

    return template


def list_template_snippets() -> dict:
    """List all available template snippets with descriptions.

    Returns:
        Dictionary of snippet names with their templates
    """
    return {
        "product_hero": "Hero product frame with floating animation and brand glow",
        "lifestyle": "Full-bleed lifestyle image with gradient overlay",
        "ai_background": "Product with AI-generated cinematic background",
        "cta_frame": "Call-to-action frame with animated button",
        "feature_frame": "Feature highlight with accent styling",
        "price_badge": "Price badge with original price, current price, and discount",
        "price_badge_simple": "Simple price badge (no discount)",
        "trust_badges": "Default trust badges (shipping, authentic, fast delivery)",
        "trust_badges_custom": "Customizable trust badges container",
        "trust_badge_item": "Single trust badge item",
        "progress_bar": "4-segment video progress bar",
        "cinematic_overlays": "Vignette, film grain, and color grade overlays",
        "reel_container": "Base reel container with overlays",
        "headline_gradient": "Gradient text headline",
        "headline_gradient_bold": "Bold gradient text headline with brand colors",
        "headline_brand": "Brand-colored headline with glow",
        "headline_highlight": "Headline with highlighted keyword",
    }


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
    # Creative direction - "auto" = smart defaults based on category + price
    video_style: str = "auto"            # auto, editorial, dynamic, product_focus, lifestyle
    mood: str = "auto"                   # auto, luxury, playful, bold, minimal
    pacing: str = "auto"                 # auto, slow, balanced, fast
    transition: str = "auto"             # auto, fade, slide, zoom, blur, wipe
    # AI Copywriting - let AI create compelling copy instead of using stale page text
    ai_copywriting: bool = True          # Generate marketing headlines (default ON)
    # Custom copy overrides (user can edit these after preview)
    custom_headline: str = ""            # Override main headline
    custom_subheadline: str = ""         # Override subheadline
    custom_cta_text: str = ""            # Override CTA button text
    custom_tagline: str = ""             # Override tagline/benefit line

class URLVideoRequest(BaseModel):
    """Generate video from URL + prompt"""
    url: str
    prompt: str = ""
    format: str = "reel"
    fps: int = 30
    record_id: str = ""  # Airtable record ID for tracking
    features: VideoFeatures = VideoFeatures()  # Feature toggles

class PreviewRequest(BaseModel):
    """Cheap preview mode - generates HTML only, skips expensive APIs"""
    url: str
    prompt: str = ""
    # Creative direction options - "auto" = smart defaults based on category + price
    video_style: str = "auto"            # auto, editorial, dynamic, product_focus, lifestyle
    mood: str = "auto"                   # auto, luxury, playful, bold, minimal
    pacing: str = "auto"                 # auto, slow, balanced, fast
    transition: str = "auto"             # auto, fade, slide, zoom, blur, wipe
    font_family: str = "Inter"
    text_color: str = "#ffffff"
    accent_color: str = ""
    # Copy customization (user editable)
    custom_headline: str = ""            # Override main headline
    custom_subheadline: str = ""         # Override subheadline
    custom_cta_text: str = ""            # Override CTA button text
    custom_tagline: str = ""             # Override tagline/benefit line

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


class VideoQuality(str, Enum):
    """Video encoding quality presets"""
    FAST = "fast"           # Quick renders, lower quality (CRF 25, preset fast)
    BALANCED = "balanced"   # Good balance of speed and quality (CRF 22, preset medium)
    PREMIUM = "premium"     # Highest quality, slower encoding (CRF 19, preset slow)


class VideoCodec(str, Enum):
    """Video codec options"""
    H264 = "h264"           # H.264/AVC - widely compatible
    H265 = "h265"           # H.265/HEVC - better quality, smaller files


# Quality preset configurations
QUALITY_PRESETS = {
    VideoQuality.FAST: {"crf": 25, "preset": "fast"},
    VideoQuality.BALANCED: {"crf": 22, "preset": "medium"},
    VideoQuality.PREMIUM: {"crf": 19, "preset": "slow"},
}


async def render_video_from_html(
    html_content: str,
    video_id: str,
    format: str = "reel",
    fps: int = 30,
    motion_blur: bool = True,
    quality: VideoQuality = VideoQuality.BALANCED,
    codec: VideoCodec = VideoCodec.H264,
    audio_url: str = None,
    audio_volume: float = 0.8,
    transition_duration: float = None
):
    """Render video from custom HTML with multiple frames

    Args:
        html_content: Custom HTML to render
        video_id: Unique identifier for the video
        format: Video format (reel, square, landscape)
        fps: Output frames per second (default 30)
        motion_blur: If True, render at 2x fps internally and blend frames for motion blur effect
        quality: Encoding quality preset (fast, balanced, premium)
        codec: Video codec (h264 or h265)
        audio_url: Optional URL to background audio track
        audio_volume: Audio volume level 0.0-1.0 (default 0.8)
        transition_duration: Override default transition duration in seconds (default 1.5)
    """
    width, height = FORMAT_DIMENSIONS.get(format, FORMAT_DIMENSIONS["reel"])

    # For motion blur, render at double fps internally
    render_fps = fps * 2 if motion_blur else fps
    output_fps = fps

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
                print(f"📊 [{video_id}] Detected {frame_count} frames (render_fps={render_fps}, output_fps={output_fps}, motion_blur={motion_blur})")

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

                        # Capture frames for this slide (at render_fps for motion blur)
                        slide_frames = int((slide_duration_ms / 1000) * render_fps)
                        for i in range(slide_frames):
                            await page.wait_for_timeout(int(1000 / render_fps))  # Wait first for stable frame
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

                        # Get per-frame transition duration or use default
                        # Check for window.transitionTiming array first, then parameter, then default
                        frame_transition = await page.evaluate(f"""(slideNum) => {{
                            if (window.transitionTiming && window.transitionTiming[slideNum]) {{
                                return window.transitionTiming[slideNum] / 1000;
                            }}
                            return null;
                        }}""", slide_num)

                        # Use priority: per-frame > parameter > default (1.5s)
                        actual_transition = frame_transition or transition_duration or 1.5
                        transition_frames = int(actual_transition * render_fps)
                        for i in range(transition_frames):
                            await page.wait_for_timeout(int(1000 / render_fps))  # Wait first, then capture
                            await page.screenshot(path=f"{output_folder}/frame_{frame_index:04d}.png")
                            frame_index += 1

                        # Clean up exit class
                        await page.evaluate(f"""(slideNum) => {{
                            document.querySelectorAll('.frame')[slideNum - 1].classList.remove('exit');
                        }}""", slide_num)

                        # Capture hold frames (remaining duration minus transition)
                        hold_frames = int((slide_duration_ms / 1000) * render_fps) - transition_frames
                        for i in range(max(0, hold_frames)):
                            await page.wait_for_timeout(int(1000 / render_fps))  # Wait first for stable frame
                            await page.screenshot(path=f"{output_folder}/frame_{frame_index:04d}.png")
                            frame_index += 1

                    video_jobs[video_id]["progress"] = int(((slide_num + 1) / frame_count) * 100)
                    print(f"✅ [{video_id}] Slide {slide_num + 1}/{frame_count}")

            else:
                duration = 5
                total_frames = render_fps * duration
                print(f"📊 [{video_id}] Single frame mode (render_fps={render_fps}, output_fps={output_fps})")

                has_seek = await page.evaluate("() => typeof window.seekToFrame === 'function'")

                for frame in range(total_frames):
                    if has_seek:
                        await page.evaluate(f"window.seekToFrame({frame}, {render_fps})")
                    else:
                        await page.wait_for_timeout(int(1000 / render_fps))
                    await page.screenshot(path=f"{output_folder}/frame_{frame_index:04d}.png")
                    frame_index += 1
                    video_jobs[video_id]["progress"] = int((frame / total_frames) * 100)

            await browser.close()

        # Compile video with quality presets, codec options, and optional audio
        video_name = f"{videos_folder}/{video_id}.mp4"
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        # Get quality settings
        quality_settings = QUALITY_PRESETS.get(quality, QUALITY_PRESETS[VideoQuality.BALANCED])
        crf = quality_settings["crf"]
        preset = quality_settings["preset"]

        # Select codec
        if codec == VideoCodec.H265:
            vcodec = "libx265"
            pix_fmt = "yuv420p10le"  # 10-bit color for H.265
            extra_params = ["-tag:v", "hvc1"]  # Apple compatibility
        else:
            vcodec = "libx264"
            pix_fmt = "yuv420p"
            extra_params = []

        print(f"🎬 [{video_id}] Encoding: quality={quality.value}, codec={codec.value}, CRF={crf}, preset={preset}")

        # Download audio if provided
        audio_path = None
        if audio_url:
            try:
                print(f"🎵 [{video_id}] Downloading audio track...")
                audio_path = os.path.join(current_folder, f"audio_{video_id}.mp3")
                async with httpx.AsyncClient(timeout=60) as client:
                    response = await client.get(audio_url)
                    if response.status_code == 200:
                        with open(audio_path, "wb") as af:
                            af.write(response.content)
                        print(f"✅ [{video_id}] Audio downloaded")
                    else:
                        print(f"⚠️ [{video_id}] Failed to download audio: {response.status_code}")
                        audio_path = None
            except Exception as e:
                print(f"⚠️ [{video_id}] Audio download error: {e}")
                audio_path = None

        # Build FFmpeg command
        if motion_blur:
            # Motion blur: blend adjacent frames (tblend) then reduce framerate
            print(f"🎬 [{video_id}] Applying motion blur (render={render_fps}fps -> output={output_fps}fps)")
            vf_filter = f"tblend=all_mode=average,fps={output_fps}"
            input_fps = render_fps
        else:
            vf_filter = None
            input_fps = output_fps

        # Base command
        cmd = [ffmpeg_exe, "-y", "-r", str(input_fps), "-i", f"{output_folder}/frame_%04d.png"]

        # Add audio input if available
        if audio_path and os.path.exists(audio_path):
            cmd.extend(["-i", audio_path])

        # Video filter
        if vf_filter:
            cmd.extend(["-vf", vf_filter])

        # Video encoding settings
        cmd.extend(["-vcodec", vcodec, "-pix_fmt", pix_fmt, "-preset", preset, "-crf", str(crf)])
        cmd.extend(extra_params)

        # Audio settings if audio is provided
        if audio_path and os.path.exists(audio_path):
            cmd.extend([
                "-c:a", "aac",
                "-b:a", "192k",
                "-filter:a", f"volume={audio_volume}",
                "-shortest"  # End video when shortest stream ends
            ])

        cmd.append(video_name)

        # Run FFmpeg
        subprocess.run(cmd)

        # Cleanup audio file
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)

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

                // All headings - but filter out UI/navigation junk
                const junkPatterns = [
                    'filter', 'sort', 'menu', 'nav', 'search', 'cart', 'login', 'sign in',
                    'account', 'wishlist', 'compare', 'category', 'categories', 'browse',
                    'shop all', 'view all', 'see all', 'show all', 'load more', 'back to',
                    'cookie', 'privacy', 'subscribe', 'newsletter', 'footer', 'header',
                    'sidebar', 'related', 'you may also', 'recently viewed', 'customers also',
                    'shipping', 'returns', 'help', 'contact', 'faq', 'about us', 'terms',
                    'select size', 'select color', 'choose', 'options', 'quantity', 'qty'
                ];
                document.querySelectorAll('h1, h2, h3').forEach(h => {
                    const text = h.textContent.trim();
                    const textLower = text.toLowerCase();
                    // Skip if too short, too long, or matches junk patterns
                    if (text && text.length > 5 && text.length < 150) {
                        const isJunk = junkPatterns.some(pattern => textLower.includes(pattern));
                        if (!isJunk) {
                            result.headings.push(text);
                        }
                    }
                });
                result.headings = result.headings.slice(0, 10);

                // Key paragraphs - filter out UI/legal/navigation text
                document.querySelectorAll('p').forEach(p => {
                    const text = p.textContent.trim();
                    const textLower = text.toLowerCase();
                    if (text && text.length > 30 && text.length < 300) {
                        const isJunk = junkPatterns.some(pattern => textLower.includes(pattern));
                        // Also skip paragraphs that look like legal/policy text
                        const isLegal = textLower.includes('©') || textLower.includes('all rights') ||
                                       textLower.includes('policy') || textLower.includes('terms of');
                        if (!isJunk && !isLegal) {
                            result.paragraphs.push(text);
                        }
                    }
                });
                result.paragraphs = result.paragraphs.slice(0, 5);

                return result;
            }""")

            extracted_copy = text_content
            print(f"📝 Extracted copy: {extracted_copy.get('product_name', 'Unknown')[:50]}...")

            # Extract images from rendered DOM - prioritize product images over branding
            images = await page.evaluate("""() => {
                const productImages = [];
                const otherImages = [];

                // Product-specific selectors (highest priority)
                const productSelectors = [
                    '.product-image img', '.product-gallery img', '.product-photo img',
                    '[data-testid*="product"] img', '[class*="product-image"] img',
                    '.gallery img', '.main-image img', '.featured-image img',
                    '[itemprop="image"]', '.woocommerce-product-gallery img',
                    '.product-single__photo img', '.product__photo img'
                ];

                // Minimum resolution for quality (400px)
                const MIN_RESOLUTION = 400;

                for (const selector of productSelectors) {
                    document.querySelectorAll(selector).forEach(img => {
                        let src = img.src || img.dataset.src || img.getAttribute('data-lazy-src');
                        if (src && src.startsWith('http') && !productImages.includes(src)) {
                            // Check resolution for product images too
                            const width = img.naturalWidth || img.width || 0;
                            const height = img.naturalHeight || img.height || 0;
                            // Accept if meets minimum resolution OR is explicitly a product URL
                            if (width >= MIN_RESOLUTION || height >= MIN_RESOLUTION || src.includes('product')) {
                                productImages.push(src);
                            }
                        }
                    });
                }

                // Get all other img elements (lower priority)
                document.querySelectorAll('img').forEach(img => {
                    let src = img.src || img.dataset.src || img.getAttribute('data-lazy-src');
                    if (src && src.startsWith('http') && !productImages.includes(src)) {
                        // Filter out tiny images, icons, tracking pixels
                        const width = img.naturalWidth || img.width || 0;
                        const height = img.naturalHeight || img.height || 0;
                        // Skip images that look like logos (wider than tall, small height)
                        const isLikelyLogo = width > height * 2 && height < 150;
                        // Increased minimum resolution for quality (400px vs old 200px)
                        if (!isLikelyLogo && (width >= MIN_RESOLUTION || height >= MIN_RESOLUTION || src.includes('product'))) {
                            otherImages.push(src);
                        }
                    }
                });

                // Get background images from divs
                document.querySelectorAll('[style*="background"]').forEach(el => {
                    const style = el.getAttribute('style');
                    const match = style.match(/url\\(['"]?(https?[^'"\\)]+)['"]?\\)/);
                    if (match && !productImages.includes(match[1])) {
                        otherImages.push(match[1]);
                    }
                });

                // og:image as fallback (often store logo, put last)
                const ogImage = document.querySelector('meta[property="og:image"]');
                if (ogImage && ogImage.content && !productImages.includes(ogImage.content)) {
                    otherImages.push(ogImage.content);
                }

                // Combine: product images first, then others
                return [...new Set([...productImages, ...otherImages])];
            }""")

            await browser.close()

            # Filter images - skip branding, logos, and UI elements
            skip_patterns = [
                'icon', 'logo', 'sprite', '.svg', 'pixel', 'tracking', '1x1', 'spacer', 'blank',
                'banner', 'header', 'footer', 'nav', 'brand', 'store-logo', 'site-logo',
                'favicon', 'badge', 'social', 'payment', 'shipping', 'trust', 'seal',
                'watermark', 'placeholder', 'loading', 'spinner', 'avatar', 'user-icon'
            ]
            for img in images:
                img_lower = img.lower()
                if any(skip in img_lower for skip in skip_patterns):
                    continue
                if img not in product_images:
                    product_images.append(img)

            product_images = product_images[:10]  # Top 10 images

    except Exception as e:
        print(f"⚠️ Playwright extraction error: {e}")

    return product_images, page_title, page_description, extracted_copy


async def generate_ai_copy(product_name: str, brand: str, price: str, description: str,
                           category: str, mood: str = "luxury") -> dict:
    """
    Use GPT to generate compelling marketing copy for the video.
    Returns headlines, taglines, and CTA text optimized for video ads.
    Cost: ~$0.01 (very small prompt)
    """
    client = OpenAI()

    # Build context from available info
    context_parts = []
    if product_name:
        context_parts.append(f"Product: {product_name}")
    if brand:
        context_parts.append(f"Brand: {brand}")
    if price:
        context_parts.append(f"Price: {price}")
    if description:
        context_parts.append(f"Description: {description[:200]}")
    if category:
        context_parts.append(f"Category: {category}")

    context = "\n".join(context_parts) if context_parts else "Generic product"

    mood_guidance = {
        "luxury": "elegant, sophisticated, aspirational",
        "playful": "fun, energetic, friendly",
        "bold": "confident, striking, powerful",
        "minimal": "clean, simple, understated"
    }
    tone = mood_guidance.get(mood, "elegant, sophisticated")

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use mini for cost efficiency (~$0.005)
            messages=[
                {"role": "system", "content": f"""You are a premium video ad copywriter. Create short, punchy marketing copy.
Tone: {tone}
Rules:
- Headlines: 2-5 words max, impactful
- Taglines: 5-8 words, benefit-focused
- CTA: 2-3 words, action-oriented
- NO generic filler like "Shop Now" for headlines
- Make it feel premium and scroll-stopping"""},
                {"role": "user", "content": f"""Create video ad copy for:
{context}

Return JSON with:
- headline: Main attention-grabbing headline (2-5 words)
- subheadline: Secondary hook or feature (3-6 words)
- tagline: Benefit statement (5-8 words)
- cta_text: Call to action button text (2-3 words)
- frame_headlines: Array of 4 short headlines for each video frame

Return ONLY valid JSON, no explanation."""}
            ],
            max_tokens=300
        )

        content = response.choices[0].message.content.strip()
        # Clean up JSON if wrapped in markdown
        content = re.sub(r'^```json?\n?', '', content)
        content = re.sub(r'\n?```$', '', content)

        copy_data = json.loads(content)
        print(f"✨ AI Copy generated: {copy_data.get('headline', 'N/A')}")
        return copy_data

    except Exception as e:
        print(f"⚠️ AI Copy generation failed: {e}")
        # Return sensible defaults
        return {
            "headline": product_name[:30] if product_name else "Discover More",
            "subheadline": "Elevate Your Style",
            "tagline": "Premium quality you can feel",
            "cta_text": "Shop Now",
            "frame_headlines": [
                product_name[:25] if product_name else "New Arrival",
                "Crafted for You",
                "Feel the Difference",
                "Get Yours Today"
            ]
        }


async def remove_background(image_url: str) -> str:
    """Remove background from product image using remove.bg API with retry logic"""
    if not REMOVE_BG_API_KEY:
        print("⚠️ No REMOVE_BG_API_KEY set, skipping background removal")
        return image_url

    async def _call_api():
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
                return f"file://{filepath}"
            elif response.status_code >= 500:
                # Server error - worth retrying
                raise Exception(f"Server error {response.status_code}: {response.text}")
            else:
                # Client error (4xx) - don't retry
                print(f"⚠️ remove.bg error: {response.status_code} - {response.text}")
                return image_url

        return image_url

    try:
        return await retry_with_backoff(_call_api, max_retries=3, base_delay=2.0)
    except Exception as e:
        print(f"⚠️ Background removal failed after retries: {e}")
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
                        'trainer', 'footwear', 'nike', 'adidas', 'jordan', 'air max',
                        'wedge', 'espadrille', 'mule', 'flat', 'pump', 'slipper', 'gia']

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

    # Party/Celebration keywords
    party_keywords = ['party', 'celebration', 'birthday', 'wedding', 'event', 'balloon',
                     'decoration', 'حفلات', 'حفلة', 'عيد', 'زفاف', 'مناسبة', 'بالون',
                     'festive', 'anniversary', 'baby shower', 'graduation', 'supplies']

    # Check categories (order matters - more specific first)
    if any(kw in text for kw in party_keywords):
        return "party"
    elif any(kw in text for kw in footwear_keywords):
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


def parse_price_value(price_str: str) -> float:
    """Extract numeric value from price string (e.g., '$149.99' -> 149.99)"""
    if not price_str:
        return 0.0
    # Remove currency symbols and whitespace, handle comma separators
    cleaned = re.sub(r'[^\d.,]', '', price_str)
    # Handle European format (1.234,56) vs US format (1,234.56)
    if ',' in cleaned and '.' in cleaned:
        if cleaned.index(',') > cleaned.index('.'):
            # European: 1.234,56
            cleaned = cleaned.replace('.', '').replace(',', '.')
        else:
            # US: 1,234.56
            cleaned = cleaned.replace(',', '')
    elif ',' in cleaned:
        # Could be European decimal or US thousands separator
        # If comma is near end with 2 digits after, treat as decimal
        if re.search(r',\d{2}$', cleaned):
            cleaned = cleaned.replace(',', '.')
        else:
            cleaned = cleaned.replace(',', '')
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def get_smart_defaults(category: str, price: str = "") -> dict:
    """
    Auto-select optimal video style, mood, pacing, and transition based on:
    - Product category (detected from title/description)
    - Price point (premium vs budget positioning)

    Returns dict with: video_style, mood, pacing, transition
    """
    price_value = parse_price_value(price)

    # Determine price tier (thresholds in USD-equivalent)
    # Premium: > $100, Mid: $30-100, Budget: < $30
    is_premium = price_value > 100
    is_budget = price_value > 0 and price_value < 30

    # Category-specific defaults optimized for best output
    category_defaults = {
        "party": {
            # Party/celebration = fun, energetic, festive
            "video_style": "dynamic",
            "mood": "playful",
            "pacing": "fast",
            "transition": "slide"
        },
        "fashion": {
            # Fashion = sophisticated, editorial, let clothes breathe
            "video_style": "editorial",
            "mood": "luxury",
            "pacing": "balanced",
            "transition": "fade"
        },
        "footwear": {
            # Footwear = streetwear energy, bold, dynamic
            "video_style": "dynamic",
            "mood": "bold",
            "pacing": "balanced",
            "transition": "slide"
        },
        "beauty": {
            # Beauty = soft, luxurious, spa-like, dreamy
            "video_style": "lifestyle",
            "mood": "luxury",
            "pacing": "slow",
            "transition": "blur"
        },
        "tech": {
            # Tech = clean, minimal, Apple-esque, product-focused
            "video_style": "product_focus",
            "mood": "minimal",
            "pacing": "balanced",
            "transition": "zoom"
        },
        "food": {
            # Food = warm, appetizing, lifestyle context
            "video_style": "lifestyle",
            "mood": "playful",
            "pacing": "slow",
            "transition": "fade"
        },
        "home": {
            # Home/furniture = architectural, minimal, lifestyle
            "video_style": "lifestyle",
            "mood": "minimal",
            "pacing": "slow",
            "transition": "fade"
        },
        "general": {
            # Safe defaults that work for most products
            "video_style": "editorial",
            "mood": "luxury",
            "pacing": "balanced",
            "transition": "fade"
        }
    }

    # Get base defaults for category
    defaults = category_defaults.get(category, category_defaults["general"]).copy()

    # Price-based adjustments
    if is_premium:
        # Premium products → slower, more luxurious feel
        if defaults["mood"] not in ["luxury", "minimal"]:
            defaults["mood"] = "luxury"
        if defaults["pacing"] == "fast":
            defaults["pacing"] = "balanced"
        # Premium fashion/beauty → slow cinematic
        if category in ["fashion", "beauty"]:
            defaults["pacing"] = "slow"
    elif is_budget:
        # Budget products → more energetic, value-focused
        if defaults["mood"] == "luxury":
            defaults["mood"] = "bold"
        if defaults["pacing"] == "slow":
            defaults["pacing"] = "balanced"
        # Budget = more dynamic energy
        if category in ["fashion", "general"]:
            defaults["video_style"] = "dynamic"
            defaults["pacing"] = "fast"

    print(f"🎯 Smart defaults for {category} (${price_value:.0f}): style={defaults['video_style']}, mood={defaults['mood']}, pacing={defaults['pacing']}")

    return defaults


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

            "party": f"""Festive celebration photography backdrop.
Elegant party atmosphere with soft bokeh lights in background.
Subtle confetti or sparkle effects, warm celebratory glow.
Clean surface with hint of festive decoration, not cluttered.
{color_hint}
Style: upscale party supply brand, celebration photography.
Warm, inviting, joyful atmosphere with premium feel.
Soft gold or silver accents, subtle shimmer textures.
NO balloons, NO party items, NO text, NO products.
ONLY the empty festive backdrop environment.
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

        async def _generate_and_download():
            response = client.images.generate(
                model="dall-e-3",
                prompt=bg_prompt,
                size="1024x1792",  # Vertical for reels
                quality="standard",
                n=1
            )

            bg_url = response.data[0].url
            print(f"✅ AI background generated")

            # Download and save locally with retry
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
                elif img_response.status_code >= 500:
                    raise Exception(f"Download failed with status {img_response.status_code}")

            return bg_url

        return await retry_with_backoff(_generate_and_download, max_retries=3, base_delay=2.0)

    except Exception as e:
        print(f"⚠️ AI background generation failed after retries: {e}")
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


def get_relative_luminance(r: int, g: int, b: int) -> float:
    """Calculate relative luminance per WCAG 2.1 specification"""
    def linearize(c):
        c = c / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * linearize(r) + 0.7152 * linearize(g) + 0.0722 * linearize(b)


def get_contrast_ratio(color1_hex: str, color2_hex: str) -> float:
    """Calculate WCAG contrast ratio between two colors

    Returns:
        Contrast ratio (1:1 to 21:1). WCAG AA requires 4.5:1 for normal text, 3:1 for large text.
        WCAG AAA requires 7:1 for normal text, 4.5:1 for large text.
    """
    r1, g1, b1 = hex_to_rgb(color1_hex)
    r2, g2, b2 = hex_to_rgb(color2_hex)

    l1 = get_relative_luminance(r1, g1, b1)
    l2 = get_relative_luminance(r2, g2, b2)

    lighter = max(l1, l2)
    darker = min(l1, l2)

    return (lighter + 0.05) / (darker + 0.05)


def check_wcag_compliance(foreground_hex: str, background_hex: str = "#0a0a0a") -> dict:
    """Check WCAG compliance for a color against background

    Args:
        foreground_hex: Text/foreground color
        background_hex: Background color (default: video dark background)

    Returns:
        Dictionary with contrast ratio and compliance levels
    """
    ratio = get_contrast_ratio(foreground_hex, background_hex)

    return {
        "contrast_ratio": round(ratio, 2),
        "aa_normal_text": ratio >= 4.5,      # 4.5:1 for normal text
        "aa_large_text": ratio >= 3.0,       # 3:1 for large text (18pt+ or 14pt bold)
        "aaa_normal_text": ratio >= 7.0,     # 7:1 for enhanced contrast
        "aaa_large_text": ratio >= 4.5,      # 4.5:1 for large text enhanced
        "passed": ratio >= 3.0,              # Minimum for large text (headlines)
        "recommendation": (
            "Excellent" if ratio >= 7.0 else
            "Good" if ratio >= 4.5 else
            "Acceptable for large text" if ratio >= 3.0 else
            "Needs adjustment"
        )
    }


def adjust_color_for_wcag(hex_color: str, background_hex: str = "#0a0a0a", target_ratio: float = 4.5) -> str:
    """Adjust a color to meet WCAG contrast requirements

    Args:
        hex_color: Color to adjust
        background_hex: Background color to contrast against
        target_ratio: Target contrast ratio (default 4.5 for AA normal text)

    Returns:
        Adjusted hex color that meets the target contrast ratio
    """
    current_ratio = get_contrast_ratio(hex_color, background_hex)

    if current_ratio >= target_ratio:
        return hex_color

    r, g, b = hex_to_rgb(hex_color)
    bg_r, bg_g, bg_b = hex_to_rgb(background_hex)
    bg_luminance = get_relative_luminance(bg_r, bg_g, bg_b)

    # For dark backgrounds, we need to lighten the color
    # For light backgrounds, we need to darken the color
    is_dark_bg = bg_luminance < 0.5

    # Iteratively adjust until we meet the target
    for i in range(100):
        if is_dark_bg:
            # Lighten
            factor = 1 + (i * 0.02)
            new_r = min(255, int(r * factor))
            new_g = min(255, int(g * factor))
            new_b = min(255, int(b * factor))
        else:
            # Darken
            factor = 1 - (i * 0.02)
            new_r = max(0, int(r * factor))
            new_g = max(0, int(g * factor))
            new_b = max(0, int(b * factor))

        new_hex = rgb_to_hex(new_r, new_g, new_b)
        if get_contrast_ratio(new_hex, background_hex) >= target_ratio:
            return new_hex

    # Fallback to white or black if adjustment fails
    return "#ffffff" if is_dark_bg else "#000000"


def validate_brand_colors(brand_colors: dict, background_hex: str = "#0a0a0a") -> dict:
    """Validate and optionally fix brand colors for WCAG compliance

    Args:
        brand_colors: Dictionary with primary, secondary, accent colors
        background_hex: Background color

    Returns:
        Dictionary with validation results and adjusted colors if needed
    """
    results = {
        "original": {},
        "adjusted": {},
        "compliance": {},
        "all_passed": True
    }

    for key in ["primary", "secondary", "accent"]:
        if key not in brand_colors:
            continue

        color = brand_colors[key]
        compliance = check_wcag_compliance(color, background_hex)
        results["original"][key] = color
        results["compliance"][key] = compliance

        if not compliance["passed"]:
            results["all_passed"] = False
            # Adjust for large text minimum (3:1) since headlines are large
            adjusted = adjust_color_for_wcag(color, background_hex, target_ratio=3.0)
            results["adjusted"][key] = adjusted
            print(f"⚠️ WCAG: {key} color {color} adjusted to {adjusted} (ratio: {compliance['contrast_ratio']} → {get_contrast_ratio(adjusted, background_hex):.2f})")
        else:
            results["adjusted"][key] = color

    return results


def get_category_color_palette(category: str) -> dict:
    """Get category-appropriate color palette instead of generic gold

    Returns a color palette optimized for the product category that creates
    better visual harmony than the default warm gold theme.
    """
    palettes = {
        "party": {
            # Festive, celebratory colors
            "primary": "#e91e63",      # Vibrant pink
            "secondary": "#9c27b0",    # Purple
            "accent": "#ff4081",       # Hot pink
            "primary_name": "vibrant pink",
            "secondary_name": "purple",
            "accent_name": "hot pink"
        },
        "fashion": {
            # Elegant, sophisticated neutrals
            "primary": "#8d6e63",      # Warm taupe
            "secondary": "#5d4037",    # Deep brown
            "accent": "#bcaaa4",       # Light taupe
            "primary_name": "warm taupe",
            "secondary_name": "deep brown",
            "accent_name": "light taupe"
        },
        "footwear": {
            # Urban, streetwear energy
            "primary": "#ff5722",      # Deep orange
            "secondary": "#bf360c",    # Burnt orange
            "accent": "#ff9800",       # Orange
            "primary_name": "deep orange",
            "secondary_name": "burnt orange",
            "accent_name": "orange"
        },
        "beauty": {
            # Soft, luxurious, spa-like
            "primary": "#f8bbd9",      # Soft pink
            "secondary": "#ce93d8",    # Lavender
            "accent": "#f48fb1",       # Rose
            "primary_name": "soft pink",
            "secondary_name": "lavender",
            "accent_name": "rose"
        },
        "tech": {
            # Clean, modern, Apple-esque
            "primary": "#42a5f5",      # Sky blue
            "secondary": "#1976d2",    # Deep blue
            "accent": "#64b5f6",       # Light blue
            "primary_name": "sky blue",
            "secondary_name": "deep blue",
            "accent_name": "light blue"
        },
        "food": {
            # Warm, appetizing, earthy
            "primary": "#8bc34a",      # Light green
            "secondary": "#689f38",    # Green
            "accent": "#ffb74d",       # Amber
            "primary_name": "light green",
            "secondary_name": "green",
            "accent_name": "amber"
        },
        "home": {
            # Neutral, warm, inviting
            "primary": "#a1887f",      # Warm grey
            "secondary": "#6d4c41",    # Brown
            "accent": "#d7ccc8",       # Light grey
            "primary_name": "warm grey",
            "secondary_name": "brown",
            "accent_name": "light grey"
        },
        "general": {
            # Original warm gold fallback for unknown categories
            "primary": "#c9a96e",      # Warm gold
            "secondary": "#8b7355",    # Warm bronze
            "accent": "#d4af37",       # Classic gold
            "primary_name": "warm gold",
            "secondary_name": "warm bronze",
            "accent_name": "classic gold"
        }
    }

    palette = palettes.get(category, palettes["general"])

    # Add original colors (same as adjusted for this case)
    return {
        **palette,
        "original_primary": palette["primary"],
        "original_secondary": palette["secondary"],
        "original_accent": palette["accent"]
    }


async def extract_colors_from_image(image_url: str, category: str = "general") -> dict:
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
                # Fallback to category-appropriate colors instead of generic gold
                category_palette = get_category_color_palette(category)
                print(f"🎨 Using category-based fallback colors for '{category}'")
                return category_palette

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
            print(f"⚠️ Extracted colors too neutral, using category-based fallback for '{category}'")
            return get_category_color_palette(category)

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


async def generate_html_from_url(url: str, prompt: str = "", features: VideoFeatures = None, return_copy_data: bool = False):
    """
    Use Playwright + OpenAI to generate HTML video content from a URL.

    Args:
        url: Product page URL
        prompt: Additional instructions
        features: VideoFeatures configuration
        return_copy_data: If True, returns (html, copy_data) tuple for preview editing

    Returns:
        str (HTML) if return_copy_data=False
        tuple (html, copy_data) if return_copy_data=True
    """

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
    # Footwear added because sandals/shoes on models are lifestyle shots
    lifestyle_categories = ["fashion", "footwear", "home", "food"]
    is_lifestyle_product = product_category in lifestyle_categories

    # Premium feature: Extract brand colors from product image (if enabled)
    # Pass product category for smarter fallback colors when extraction fails
    brand_colors = None
    if features.color_extraction and product_images:
        brand_colors = await extract_colors_from_image(product_images[0], category=product_category)

        # Validate and adjust colors for WCAG compliance
        if brand_colors:
            wcag_results = validate_brand_colors(brand_colors)
            if not wcag_results["all_passed"]:
                print(f"🎨 WCAG: Adjusting colors for accessibility compliance")
                brand_colors["primary"] = wcag_results["adjusted"].get("primary", brand_colors["primary"])
                brand_colors["secondary"] = wcag_results["adjusted"].get("secondary", brand_colors["secondary"])
                brand_colors["accent"] = wcag_results["adjusted"].get("accent", brand_colors["accent"])

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

    # AI Copywriting: Generate compelling marketing copy
    ai_copy = None
    if features.ai_copywriting:
        print(f"✨ Generating AI marketing copy...")
        ai_copy = await generate_ai_copy(
            product_name=extracted_copy.get("product_name", ""),
            brand=extracted_copy.get("brand", ""),
            price=extracted_copy.get("price", ""),
            description=extracted_copy.get("description", ""),
            category=product_category,
            mood=features.mood
        )

    # Apply custom copy overrides (user can edit these after preview)
    final_copy = {
        "headline": features.custom_headline or (ai_copy.get("headline") if ai_copy else ""),
        "subheadline": features.custom_subheadline or (ai_copy.get("subheadline") if ai_copy else ""),
        "tagline": features.custom_tagline or (ai_copy.get("tagline") if ai_copy else ""),
        "cta_text": features.custom_cta_text or (ai_copy.get("cta_text") if ai_copy else "Shop Now"),
        "frame_headlines": ai_copy.get("frame_headlines", []) if ai_copy else []
    }

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
        # Use category-appropriate colors instead of generic gold
        category_palette = get_category_color_palette(product_category)
        primary_color = category_palette["primary"]
        secondary_color = category_palette["secondary"]
        accent_color = category_palette["accent"]
        primary_rgb = hex_to_rgb(primary_color)
        secondary_rgb = hex_to_rgb(secondary_color)
        accent_rgb = hex_to_rgb(accent_color)
        color_description = f"Using {product_category} category palette: {category_palette['primary_name']} ({primary_color}), {category_palette['secondary_name']} ({secondary_color}), {category_palette['accent_name']} ({accent_color})"

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

    # Creative direction settings - use smart defaults when "auto"
    smart_defaults = get_smart_defaults(product_category, extracted_copy.get("price", ""))

    video_style = smart_defaults["video_style"] if features.video_style == "auto" else features.video_style
    mood = smart_defaults["mood"] if features.mood == "auto" else features.mood
    pacing = smart_defaults["pacing"] if features.pacing == "auto" else features.pacing
    transition = smart_defaults["transition"] if features.transition == "auto" else features.transition

    print(f"🎬 Creative: Style={video_style}, Mood={mood}, Pacing={pacing}, Transition={transition}")

    # Video style templates - proven composition structures
    style_templates = {
        "editorial": """EDITORIAL STYLE (Fashion Magazine Aesthetic):
- Frame 1: Hero lifestyle shot - model/product fills frame, dramatic lighting
- Frame 2: Detail or alternate angle - focus on craftsmanship/texture
- Frame 3: Lifestyle context - product in use/worn
- Frame 4: CTA frame - clean product shot with button
COMPOSITION: Full-bleed images, minimal text, let visuals breathe. One strong headline per frame.""",

        "dynamic": """DYNAMIC STYLE (High Energy):
- Frame 1: Bold opening - product bursting onto screen
- Frame 2: Quick feature highlight - zoom on key detail
- Frame 3: Action/movement shot - energy and motion
- Frame 4: Punchy CTA - urgent, compelling call to action
COMPOSITION: Strong angles, dynamic crops, impactful text. Fast visual rhythm.""",

        "product_focus": """PRODUCT FOCUS STYLE (Hero Shots):
- Frame 1: Hero product shot - centered, prominent, glowing
- Frame 2: Feature callout - highlight key benefit with text
- Frame 3: Social proof or trust - badges, reviews sentiment
- Frame 4: CTA with product - final beauty shot + button
COMPOSITION: Product is the star. Clean backgrounds, product fills 70% of frame.""",

        "lifestyle": """LIFESTYLE STYLE (Contextual Storytelling):
- Frame 1: Scene setting - environment/mood establishing shot
- Frame 2: Product in context - natural usage scenario
- Frame 3: Emotional benefit - how it makes you feel
- Frame 4: Aspirational CTA - "Join the lifestyle" feeling
COMPOSITION: Full environmental shots, product integrated naturally, storytelling focus."""
    }

    # Mood modifiers
    mood_modifiers = {
        "luxury": "MOOD: Elegant, sophisticated, premium. Use subtle animations, refined typography, muted color transitions. Think Chanel, Rolex, high-end fashion.",
        "playful": "MOOD: Fun, energetic, youthful. Bright accents OK, bouncy animations, friendly copy. Think Nike, Glossier, modern DTC brands.",
        "bold": "MOOD: Striking, confident, unapologetic. Strong contrasts, impactful headlines, powerful imagery. Think Supreme, Beats, athletic brands.",
        "minimal": "MOOD: Clean, refined, understated. Maximum whitespace, simple typography, subtle effects. Think Apple, Muji, Scandinavian design."
    }

    # Pacing/timing presets (in milliseconds per frame)
    # First frame is longer to let viewers read the opening headline
    pacing_timings = {
        "slow": [5500, 4500, 4500, 5000],      # Cinematic, let it breathe (first frame +1s)
        "balanced": [4500, 3500, 3500, 4000],   # Standard rhythm (first frame +1s)
        "fast": [3200, 2500, 2500, 3000]        # Quick cuts, energy (first frame +0.7s)
    }

    # Transition styles - CSS for different frame transitions
    transition_styles = {
        "fade": """/* FADE TRANSITION - Smooth crossfade */
.frame { position: absolute; top: 0; left: 0; width: 100%; height: 100%; opacity: 0; transition: opacity 1.2s ease-in-out; }
.frame.active { opacity: 1; }
.frame.exit { opacity: 0; }""",

        "slide": """/* SLIDE TRANSITION - Horizontal slide */
.frame { position: absolute; top: 0; left: 0; width: 100%; height: 100%; opacity: 0; transform: translateX(100%); transition: transform 0.8s ease-in-out, opacity 0.8s ease-in-out; }
.frame.active { opacity: 1; transform: translateX(0); }
.frame.exit { opacity: 0; transform: translateX(-100%); }""",

        "zoom": """/* ZOOM TRANSITION - Scale in/out */
.frame { position: absolute; top: 0; left: 0; width: 100%; height: 100%; opacity: 0; transform: scale(0.8); transition: transform 1s ease-out, opacity 1s ease-out; }
.frame.active { opacity: 1; transform: scale(1); }
.frame.exit { opacity: 0; transform: scale(1.2); }""",

        "blur": """/* BLUR TRANSITION - Focus/defocus effect */
.frame { position: absolute; top: 0; left: 0; width: 100%; height: 100%; opacity: 0; filter: blur(20px); transition: opacity 1s ease-in-out, filter 1s ease-in-out; }
.frame.active { opacity: 1; filter: blur(0); }
.frame.exit { opacity: 0; filter: blur(20px); }""",

        "wipe": """/* WIPE TRANSITION - Reveal effect */
.frame { position: absolute; top: 0; left: 0; width: 100%; height: 100%; opacity: 0; clip-path: inset(0 100% 0 0); transition: clip-path 1s ease-in-out, opacity 0.5s ease-in-out; }
.frame.active { opacity: 1; clip-path: inset(0 0 0 0); }
.frame.exit { opacity: 0; clip-path: inset(0 0 0 100%); }"""
    }

    style_instruction = style_templates.get(video_style, style_templates["editorial"])
    mood_instruction = mood_modifiers.get(mood, mood_modifiers["luxury"])
    timing_values = pacing_timings.get(pacing, pacing_timings["balanced"])
    transition_css = transition_styles.get(transition, transition_styles["fade"])

    system_prompt = f"""You are a premium video ad designer. Create cinematic Instagram Reel HTML videos.

🎬 CREATIVE DIRECTION:
{style_instruction}

{mood_instruction}

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
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family={font_url_name}:wght@400;700;900&display=swap" rel="stylesheet">
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}

/* FONT SMOOTHING & TEXT RENDERING */
html {{
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-rendering: optimizeLegibility;
}}

/* ANIMATED GRADIENT BACKGROUND - Using brand colors */
body {{ background: #0a0a0a; -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; text-rendering: optimizeLegibility; }}
.reel-container {{
  width: 1080px; height: 1920px; position: relative; overflow: hidden;
  background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #0a0a0a 100%);
}}

/* CINEMATIC VIGNETTE - Subtle edge darkening */
.vignette {{
  position: absolute; top: 0; left: 0; width: 100%; height: 100%;
  background: radial-gradient(ellipse at center, transparent 60%, rgba(0,0,0,0.25) 100%);
  z-index: 50; pointer-events: none;
}}

/* FILM GRAIN OVERLAY - Visible texture for premium look */
.film-grain {{
  position: absolute; top: 0; left: 0; width: 100%; height: 100%;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E");
  opacity: 0.05; z-index: 51; pointer-events: none; mix-blend-mode: overlay;
  animation: grainShift 0.5s steps(1) infinite;
}}
/* Animated grain for authentic film look */
@keyframes grainShift {{
  0%, 100% {{ transform: translate(0, 0); }}
  25% {{ transform: translate(-2px, 2px); }}
  50% {{ transform: translate(2px, -2px); }}
  75% {{ transform: translate(-2px, -2px); }}
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

{transition_css}
.frame {{ display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 0; }}

/* Safe zone helper - balanced vertical distribution */
.safe-zone {{ position: absolute; top: 200px; left: 80px; right: 180px; bottom: 350px; display: flex; flex-direction: column; align-items: center; justify-content: space-between; }}
.content-area {{ flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; width: 100%; }}

/* PREMIUM EASING - Natural motion curves */
:root {{
  --ease-out-expo: cubic-bezier(0.16, 1, 0.3, 1);
  --ease-out-quint: cubic-bezier(0.22, 1, 0.36, 1);
  --ease-in-out-smooth: cubic-bezier(0.4, 0, 0.2, 1);
}}

/* PRODUCT ANIMATIONS - Staggered entrances with premium easing */
.frame.active .product-wrap {{ animation: floatIn 1.2s var(--ease-out-expo) forwards, float 10s var(--ease-in-out-smooth) 1.5s infinite; }}
.frame.active .text-area {{ animation: fadeUp 0.8s var(--ease-out-expo) 0.5s forwards; opacity: 0; }}
.frame.active .text-area h1 {{ animation: fadeUp 0.7s var(--ease-out-expo) 0.5s forwards; opacity: 0; }}
.frame.active .text-area p {{ animation: fadeUp 0.7s var(--ease-out-expo) 0.7s forwards; opacity: 0; }}
.frame.active .accent-line {{ animation: lineGrow 0.6s var(--ease-out-quint) 0.9s forwards; }}

/* LIFESTYLE - Override: NO animations at all */
.frame.lifestyle.active .lifestyle-img {{ animation: none !important; transform: none !important; }}
.frame.lifestyle.active .text-area {{ animation: fadeUp 0.8s var(--ease-out-expo) 0.2s forwards; opacity: 0; }}
.frame.lifestyle.active .text-area h1 {{ animation: fadeUp 0.7s var(--ease-out-expo) 0.3s forwards; opacity: 0; }}
.frame.lifestyle.active .text-area p {{ animation: fadeUp 0.7s var(--ease-out-expo) 0.5s forwards; opacity: 0; }}

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
.ai-bg {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; opacity: 1; z-index: 0; }}
.frame.ai-background .ai-bg {{ animation: none; }}
/* Gradient overlay to push AI background back and make product pop */
.ai-bg-overlay {{
  position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 1;
  background: radial-gradient(ellipse at center, rgba(0,0,0,0.2) 0%, rgba(0,0,0,0.5) 70%, rgba(0,0,0,0.7) 100%);
  pointer-events: none;
}}

/* PREMIUM TEXT STYLING - 15% from bottom, properly centered */
/* Symmetric margins for true center alignment */
.text-area {{
  position: absolute; bottom: 300px; left: 80px; right: 80px;
  text-align: center; z-index: 10;
  /* Overflow prevention */
  max-width: calc(100% - 160px);
  overflow: hidden;
}}
h1 {{
  font-family: '{font_family}', sans-serif;
  font-size: clamp(48px, 7vw, 72px); /* Auto-scales based on container */
  font-weight: 900;
  color: {text_color}; text-transform: uppercase; line-height: 1.1; letter-spacing: -1px;
  text-shadow: 0 4px 30px rgba(0,0,0,0.5), 0 0 60px rgba({primary_rgb[0]},{primary_rgb[1]},{primary_rgb[2]},0.3);
  word-wrap: break-word;
  overflow-wrap: break-word;
  hyphens: auto;
}}
/* Smaller headline variant for long text */
h1.headline-sm {{ font-size: clamp(36px, 5vw, 56px); }}
h1.headline-lg {{ font-size: clamp(56px, 8vw, 84px); }}

/* TEXT OVERFLOW PREVENTION - Clean truncation with ellipsis */
/* Single line ellipsis */
.text-ellipsis {{
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 100%;
}}
/* Multi-line clamp with ellipsis - 2 lines */
.text-clamp-2 {{
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  text-overflow: ellipsis;
}}
/* Multi-line clamp with ellipsis - 3 lines */
.text-clamp-3 {{
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
  text-overflow: ellipsis;
}}
/* Multi-line clamp with ellipsis - 4 lines */
.text-clamp-4 {{
  display: -webkit-box;
  -webkit-line-clamp: 4;
  -webkit-box-orient: vertical;
  overflow: hidden;
  text-overflow: ellipsis;
}}
/* Safe text area that won't overflow container */
.text-safe {{
  max-height: 400px;
  overflow: hidden;
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
p {{
  font-family: '{font_family}', sans-serif; font-size: 32px; font-weight: 400;
  color: {text_color}; opacity: 0.7; margin-top: 16px; letter-spacing: 1px;
  text-shadow: 0 2px 20px rgba(0,0,0,0.6);
  /* Overflow prevention */
  word-wrap: break-word;
  overflow-wrap: break-word;
  max-width: 100%;
}}
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
  animation: ctaAppear 0.8s cubic-bezier(0.34, 1.56, 0.64, 1) 1s forwards, ctaFloat 6s var(--ease-in-out-smooth) 1.8s infinite;
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
  50% {{ transform: translateY(-4px); }}
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
  animation: trustAppear 0.6s var(--ease-out-expo) 1.2s forwards;
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
  50% {{ transform: translateY(-3px); }}
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

📝 TEXT OVERFLOW PREVENTION - ALWAYS use for long text:
   - class='text-ellipsis' - single line truncation with ellipsis (...)
   - class='text-clamp-2' - limit to 2 lines with ellipsis
   - class='text-clamp-3' - limit to 3 lines with ellipsis
   - class='text-clamp-4' - limit to 4 lines with ellipsis
   - class='text-safe' - container that won't overflow (max-height with hidden overflow)
   RULES: Headlines should use text-clamp-2, subtitles text-clamp-3. NEVER let text overflow frame!

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
  <div class="ai-bg-overlay"></div>
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

Add at end: <script>const timing = {timing_values};</script>

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

    # Add AI-generated copy if available
    if final_copy.get("headline") or final_copy.get("frame_headlines"):
        user_prompt_parts.extend([
            "",
            "✨ AI-GENERATED MARKETING COPY (USE THESE EXACT HEADLINES):",
        ])
        if final_copy.get("headline"):
            user_prompt_parts.append(f"   MAIN HEADLINE: {final_copy['headline']}")
        if final_copy.get("subheadline"):
            user_prompt_parts.append(f"   SUBHEADLINE: {final_copy['subheadline']}")
        if final_copy.get("tagline"):
            user_prompt_parts.append(f"   TAGLINE: {final_copy['tagline']}")
        if final_copy.get("cta_text"):
            user_prompt_parts.append(f"   CTA BUTTON TEXT: {final_copy['cta_text']}")
        if final_copy.get("frame_headlines"):
            user_prompt_parts.append("   FRAME-BY-FRAME HEADLINES:")
            for i, headline in enumerate(final_copy["frame_headlines"][:4]):
                frame_num = i + 1
                user_prompt_parts.append(f"      Frame {frame_num}: {headline}")
        user_prompt_parts.extend([
            "",
            "⚠️ USE THE HEADLINES ABOVE - They are professionally crafted for this product.",
            "DO NOT make up your own headlines - use the AI-generated ones provided.",
        ])

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
            "🚨🚨🚨 LIFESTYLE MODE - ABSOLUTE RULES (VIOLATION = FAILURE) 🚨🚨🚨",
            "",
            "⛔ NEVER DO THESE (INSTANT FAIL):",
            "- NEVER put multiple images in one frame",
            "- NEVER make images small or resize them",
            "- NEVER add backgrounds behind images",
            "- NEVER use .product-wrap or .product-img classes",
            "- NEVER use .bg-glow or .ai-bg",
            "- NEVER composite or collage images together",
            "",
            "✅ MUST DO (REQUIRED):",
            "1. ONE image per frame, DIFFERENT image each frame",
            "2. Image fills 100% width and 100% height (object-fit: cover)",
            "3. Use ONLY .lifestyle-img class on images",
            "4. Simple text overlay at bottom with .lifestyle-overlay gradient",
            "5. CTA button ONLY on final frame",
            "",
            "CORRECT HTML (copy this EXACTLY):",
            '<div class="frame lifestyle active">',
            '  <img src="IMAGE_URL_HERE" class="lifestyle-img">',
            '  <div class="lifestyle-overlay"></div>',
            '  <div class="text-area"><h1>Simple Headline</h1></div>',
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

    # Return copy data for preview editing if requested
    if return_copy_data:
        copy_data = {
            "headline": final_copy.get("headline", ""),
            "subheadline": final_copy.get("subheadline", ""),
            "tagline": final_copy.get("tagline", ""),
            "cta_text": final_copy.get("cta_text", "Shop Now"),
            "frame_headlines": final_copy.get("frame_headlines", []),
            "product_name": extracted_copy.get("product_name", ""),
            "price": extracted_copy.get("price", ""),
            "brand": extracted_copy.get("brand", ""),
            "category": product_category,
            # Include the resolved creative settings (useful when "auto" was used)
            "creative_settings": {
                "video_style": video_style,
                "mood": mood,
                "pacing": pacing,
                "transition": transition
            }
        }
        return html_content, copy_data

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

        /* Creative direction */
        .creative-section { margin: 25px 0; padding: 20px; background: #111; border-radius: 12px; border: 1px solid #222; }
        .creative-section h3 { font-size: 16px; margin-bottom: 20px; color: #ccc; display: flex; align-items: center; gap: 8px; }
        .creative-section h3::before { content: '🎬'; }
        .creative-grid { display: flex; flex-direction: column; gap: 20px; }
        .creative-group label:first-child { display: block; font-size: 13px; color: #888; margin-bottom: 10px; }

        /* Radio cards for video style */
        .radio-cards { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
        .radio-card { cursor: pointer; }
        .radio-card input { display: none; }
        .radio-card .card-content { display: flex; flex-direction: column; align-items: center; padding: 15px 10px; background: #1a1a1a; border: 2px solid #333; border-radius: 12px; transition: all 0.2s; text-align: center; }
        .radio-card:hover .card-content { border-color: #444; background: #222; }
        .radio-card input:checked + .card-content { border-color: #6366f1; background: rgba(99,102,241,0.1); }
        .radio-card .card-icon { font-size: 24px; margin-bottom: 8px; }
        .radio-card .card-title { font-size: 14px; font-weight: 600; color: white; margin-bottom: 4px; }
        .radio-card .card-desc { font-size: 11px; color: #888; }

        /* Pill options for mood/pacing */
        .pill-options { display: flex; flex-wrap: wrap; gap: 8px; }
        .pill-option { cursor: pointer; }
        .pill-option input { display: none; }
        .pill-option span { display: inline-block; padding: 10px 16px; background: #1a1a1a; border: 1px solid #333; border-radius: 25px; font-size: 13px; color: #aaa; transition: all 0.2s; }
        .pill-option:hover span { border-color: #444; color: white; }
        .pill-option input:checked + span { border-color: #6366f1; background: rgba(99,102,241,0.15); color: white; }

        @media (max-width: 600px) {
            .features-grid { grid-template-columns: 1fr; }
            .style-row { flex-direction: column; gap: 15px; }
            .radio-cards { grid-template-columns: 1fr; }
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

            <div class="creative-section">
                <h3>Creative Direction</h3>
                <div class="creative-grid">
                    <div class="creative-group">
                        <label>Video Style</label>
                        <div class="radio-cards">
                            <label class="radio-card">
                                <input type="radio" name="videoStyle" value="editorial" checked>
                                <span class="card-content">
                                    <span class="card-icon">📸</span>
                                    <span class="card-title">Editorial</span>
                                    <span class="card-desc">Fashion magazine feel</span>
                                </span>
                            </label>
                            <label class="radio-card">
                                <input type="radio" name="videoStyle" value="dynamic">
                                <span class="card-content">
                                    <span class="card-icon">⚡</span>
                                    <span class="card-title">Dynamic</span>
                                    <span class="card-desc">Fast cuts, energy</span>
                                </span>
                            </label>
                            <label class="radio-card">
                                <input type="radio" name="videoStyle" value="product_focus">
                                <span class="card-content">
                                    <span class="card-icon">🎯</span>
                                    <span class="card-title">Product Focus</span>
                                    <span class="card-desc">Hero product shots</span>
                                </span>
                            </label>
                            <label class="radio-card">
                                <input type="radio" name="videoStyle" value="lifestyle">
                                <span class="card-content">
                                    <span class="card-icon">🌟</span>
                                    <span class="card-title">Lifestyle</span>
                                    <span class="card-desc">Contextual scenes</span>
                                </span>
                            </label>
                        </div>
                    </div>
                    <div class="creative-group">
                        <label>Mood</label>
                        <div class="pill-options">
                            <label class="pill-option"><input type="radio" name="mood" value="luxury" checked><span>✨ Luxury</span></label>
                            <label class="pill-option"><input type="radio" name="mood" value="playful"><span>🎈 Playful</span></label>
                            <label class="pill-option"><input type="radio" name="mood" value="bold"><span>💥 Bold</span></label>
                            <label class="pill-option"><input type="radio" name="mood" value="minimal"><span>◽ Minimal</span></label>
                        </div>
                    </div>
                    <div class="creative-group">
                        <label>Pacing</label>
                        <div class="pill-options">
                            <label class="pill-option"><input type="radio" name="pacing" value="slow"><span>🎬 Slow & Cinematic</span></label>
                            <label class="pill-option"><input type="radio" name="pacing" value="balanced" checked><span>⚖️ Balanced</span></label>
                            <label class="pill-option"><input type="radio" name="pacing" value="fast"><span>🚀 Fast & Energetic</span></label>
                        </div>
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
                accent_color: document.getElementById('accentColor').value,
                // Creative direction
                video_style: document.querySelector('input[name="videoStyle"]:checked').value,
                mood: document.querySelector('input[name="mood"]:checked').value,
                pacing: document.querySelector('input[name="pacing"]:checked').value
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

                if (!res.ok) {
                    throw new Error(data.detail || data.error || 'Request failed');
                }

                if (data.video_id) {
                    currentMultiFormat = data.multi_format || false;
                    currentFormatIds = data.format_ids || null;
                    pollStatus(data.video_id);
                } else {
                    throw new Error(data.detail || 'Failed to start');
                }
            } catch (err) {
                const errorMsg = err.message || 'An unexpected error occurred';
                statusText.innerHTML = '<span class="error">Error: ' + errorMsg + '</span>';
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

                // Handle HTTP errors (4xx, 5xx)
                if (!res.ok) {
                    const errorMsg = data.detail || data.error || 'Request failed';
                    statusText.innerHTML = '<span class="error">Error: ' + errorMsg + '</span>';
                    submitBtn.disabled = false;
                    submitBtn.textContent = 'Try Again';
                    return;
                }

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
                    const errorMsg = data.error || 'Unknown error occurred';
                    statusText.innerHTML = '<span class="error">Error: ' + errorMsg + '</span>';
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

@app.post("/preview-from-url")
async def preview_from_url(request: PreviewRequest, _: bool = Depends(verify_api_key)):
    """
    CHEAP PREVIEW MODE - Returns HTML + editable copy, no video rendering.

    Cost: ~$0.03-0.05 (GPT-4o + GPT-4o-mini for copy)
    Skips: DALL-E backgrounds (~$0.04), Remove.bg (~$0.10)

    Returns:
    - HTML preview for iframe display
    - AI-generated marketing copy (editable)
    - User can edit copy, then call /generate-from-url with custom_* fields

    Use this to let users preview and customize before full video render.
    """
    preview_id = str(uuid.uuid4())[:8]

    try:
        print(f"👁️ [{preview_id}] PREVIEW MODE - Analyzing URL: {request.url}")

        # Create features object with expensive APIs disabled
        preview_features = VideoFeatures(
            background_removal=False,      # Skip Remove.bg (~$0.10 saved)
            ai_background=False,           # Skip DALL-E (~$0.04 saved)
            color_extraction=True,         # Keep - uses PIL, free
            cta_button=True,
            progress_bar=True,
            text_effects=True,
            floating_animation=True,
            ken_burns=True,
            smart_copy=True,
            price_badge=True,
            trust_badges=True,
            multi_format=False,
            ai_copywriting=True,           # Generate AI marketing copy
            # User's creative direction choices
            font_family=request.font_family,
            text_color=request.text_color,
            accent_color=request.accent_color,
            video_style=request.video_style,
            mood=request.mood,
            pacing=request.pacing,
            transition=request.transition,
            # Pass any custom copy overrides
            custom_headline=request.custom_headline,
            custom_subheadline=request.custom_subheadline,
            custom_cta_text=request.custom_cta_text,
            custom_tagline=request.custom_tagline
        )

        # Generate HTML with copy data for editing
        html_content, copy_data = await generate_html_from_url(
            request.url, request.prompt, preview_features, return_copy_data=True
        )
        print(f"✅ [{preview_id}] Preview HTML generated ({len(html_content)} chars)")
        print(f"✨ [{preview_id}] AI Copy: {copy_data.get('headline', 'N/A')}")

        # Store preview for retrieval
        video_jobs[preview_id] = {
            "status": "preview_complete",
            "html": html_content,
            "url": request.url,
            "copy_data": copy_data
        }

        # Return HTML + editable copy fields
        return {
            "message": "Preview generated successfully",
            "status": "complete",
            "preview_id": preview_id,
            "html": html_content,
            "preview_url": f"/preview/{preview_id}",
            "url": request.url,
            "estimated_cost": "$0.03-0.05",
            # Editable copy fields - user can modify these and resubmit
            "copy": {
                "headline": copy_data.get("headline", ""),
                "subheadline": copy_data.get("subheadline", ""),
                "tagline": copy_data.get("tagline", ""),
                "cta_text": copy_data.get("cta_text", "Shop Now"),
                "frame_headlines": copy_data.get("frame_headlines", []),
            },
            # Product info for context
            "product": {
                "name": copy_data.get("product_name", ""),
                "price": copy_data.get("price", ""),
                "brand": copy_data.get("brand", ""),
                "category": copy_data.get("category", "")
            },
            # Auto-selected creative direction (based on category + price)
            "creative_settings": copy_data.get("creative_settings", {
                "video_style": request.video_style,
                "mood": request.mood,
                "pacing": request.pacing,
                "transition": request.transition
            }),
            "note": "Edit the 'copy' fields above, then call /generate-from-url with custom_headline, custom_subheadline, etc. to render video with your edits."
        }

    except Exception as e:
        print(f"❌ [{preview_id}] Preview error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/preview/{preview_id}", response_class=HTMLResponse)
async def get_preview(preview_id: str):
    """Serve preview HTML directly for iframe embedding"""
    if preview_id not in video_jobs:
        raise HTTPException(status_code=404, detail="Preview not found")

    job = video_jobs[preview_id]
    if job.get("status") != "preview_complete":
        raise HTTPException(status_code=404, detail="Preview not found")

    return HTMLResponse(content=job["html"], status_code=200)

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


# --- COMPONENT SYSTEM API ---

@app.get("/components")
async def list_components(category: str = None, tag: str = None):
    """List all available components with full metadata

    Query params:
        category: Filter by category (frame, element, text, overlay, container)
        tag: Filter by tag (e.g., 'hero', 'cta', 'premium')
    """
    cat = ComponentCategory(category) if category else None
    tags = [tag] if tag else None
    components = component_registry.list(category=cat, tags=tags)

    return {
        "count": len(components),
        "components": [c.to_dict() for c in components],
        "categories": [c.value for c in ComponentCategory],
        "all_tags": list(set(t for c in component_registry.list() for t in c.tags))
    }


@app.get("/components/{name}")
async def get_component(name: str):
    """Get full details of a specific component"""
    try:
        component = component_registry.get(name)
        return component.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


class ComponentRenderRequest(BaseModel):
    """Request to render a component"""
    variant: Optional[str] = None
    props: dict = {}


@app.post("/components/{name}/render")
async def render_component(name: str, request: ComponentRenderRequest):
    """Render a component with props and optional variant

    Example:
        POST /components/product_hero/render
        {
            "variant": "with_price",
            "props": {
                "image_url": "https://example.com/shoe.png",
                "headline": "Step Into Style",
                "price": "$129"
            }
        }
    """
    try:
        html = component_registry.render(name, variant=request.variant, **request.props)
        component = component_registry.get(name)
        return {
            "name": name,
            "variant": request.variant,
            "html": html,
            "props_used": list(request.props.keys()),
            "available_variants": list(component.variants.keys())
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# --- VIDEO PRESETS API ---

@app.get("/presets")
async def list_presets():
    """List all available video presets"""
    return {
        "presets": VIDEO_PRESETS,
        "usage": "Use POST /presets/{name}/build to generate video HTML from a preset"
    }


@app.get("/presets/{name}")
async def get_preset(name: str):
    """Get details of a specific preset"""
    if name not in VIDEO_PRESETS:
        raise HTTPException(
            status_code=404,
            detail=f"Preset '{name}' not found. Available: {list(VIDEO_PRESETS.keys())}"
        )
    preset = VIDEO_PRESETS[name]
    return {
        **preset,
        "frame_components": [component_registry.get(f).to_dict() for f in preset["frames"]]
    }


class PresetBuildRequest(BaseModel):
    """Request to build a video from a preset"""
    images: List[str]
    headlines: List[str]
    subheadline: str = ""
    product_name: str = "Product"
    cta_text: str = "Shop Now"
    extra_props: dict = {}


@app.post("/presets/{name}/build")
async def build_from_preset(name: str, request: PresetBuildRequest):
    """Build a video from a preset template

    Example:
        POST /presets/product_showcase/build
        {
            "images": ["https://example.com/img1.png", "https://example.com/img2.png"],
            "headlines": ["Step Into Style", "Premium Comfort", "Made to Last", "Shop Now"],
            "subheadline": "Elevate your look",
            "product_name": "Nike Air Max",
            "cta_text": "Get Yours"
        }
    """
    try:
        builder = get_video_builder()
        builder.set_props(
            subheadline=request.subheadline,
            product_name=request.product_name,
            cta_text=request.cta_text,
            feature_text=request.subheadline,
            **request.extra_props
        )
        builder.from_preset(name, request.images, request.headlines)
        result = builder.build()

        return {
            "preset": name,
            "html": result["html"],
            "timing": result["timing"],
            "frame_count": result["frame_count"],
            "timing_script": f"<script>window.timing = {result['timing']};</script>"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# --- VIDEO BUILDER API ---

class VideoFrame(BaseModel):
    """A single frame in a custom video"""
    component: str
    variant: Optional[str] = None
    duration_ms: int = 4000
    props: dict = {}


class VideoBuildRequest(BaseModel):
    """Request to build a custom video"""
    frames: List[VideoFrame]
    global_props: dict = {}


@app.post("/video/build")
async def build_custom_video(request: VideoBuildRequest):
    """Build a custom video by specifying frames and components

    Example:
        POST /video/build
        {
            "global_props": {
                "product_name": "Nike Air Max",
                "subheadline": "Premium comfort"
            },
            "frames": [
                {
                    "component": "product_hero",
                    "duration_ms": 5000,
                    "props": {
                        "image_url": "https://example.com/img1.png",
                        "headline": "Step Into Style"
                    }
                },
                {
                    "component": "feature_frame",
                    "duration_ms": 4000,
                    "props": {
                        "image_url": "https://example.com/img2.png",
                        "headline": "Comfort First",
                        "feature_text": "All-day cushioning"
                    }
                },
                {
                    "component": "cta_frame",
                    "variant": "with_trust",
                    "duration_ms": 4500,
                    "props": {
                        "image_url": "https://example.com/img1.png",
                        "headline": "Get Yours Today",
                        "cta_text": "Shop Now"
                    }
                }
            ]
        }
    """
    try:
        builder = get_video_builder()
        builder.set_props(**request.global_props)

        for frame in request.frames:
            builder.add_frame(
                frame.component,
                variant=frame.variant,
                duration_ms=frame.duration_ms,
                **frame.props
            )

        result = builder.build()

        return {
            "html": result["html"],
            "timing": result["timing"],
            "frame_count": result["frame_count"],
            "timing_script": f"<script>window.timing = {result['timing']};</script>"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# --- LEGACY TEMPLATE API (backwards compatible) ---

@app.get("/templates")
async def get_templates():
    """List all available template snippets (legacy API)"""
    return {
        "snippets": list_template_snippets(),
        "usage": "Use GET /templates/{name} to get a specific template, or POST /templates/{name} with JSON body to get a filled template",
        "note": "Consider using the new /components API for more features"
    }


@app.get("/templates/{name}")
async def get_template(name: str):
    """Get a specific template snippet by name (legacy API)"""
    if name not in TEMPLATE_SNIPPETS:
        raise HTTPException(
            status_code=404,
            detail=f"Template '{name}' not found. Available: {list(TEMPLATE_SNIPPETS.keys())}"
        )
    return {
        "name": name,
        "template": TEMPLATE_SNIPPETS[name],
        "description": list_template_snippets().get(name, "")
    }


class TemplateRequest(BaseModel):
    """Request body for filling a template with variables"""
    variables: dict = {}


@app.post("/templates/{name}")
async def fill_template(name: str, request: TemplateRequest):
    """Fill a template snippet with provided variables (legacy API)

    Example:
        POST /templates/product_hero
        {
            "variables": {
                "image_url": "https://example.com/shoe.png",
                "product_name": "Nike Air Max",
                "headline": "Step Into Style",
                "subheadline": "Comfort meets design"
            }
        }
    """
    try:
        html = get_template_snippet(name, **request.variables)
        return {
            "name": name,
            "html": html,
            "variables_used": list(request.variables.keys())
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
