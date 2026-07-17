"""
AI Interior Design Generator
-----------------------------
A Streamlit app that lets a user pick a room from a house floor plan,
choose design parameters (style, material, flooring, windows/doors,
color palette, lighting) and generates a photorealistic interior
render using a Hugging Face text-to-image model via the Inference API.

Deploy on Streamlit Community Cloud or run locally:
    streamlit run app.py

Set your Hugging Face token either:
  1. In Streamlit secrets:  .streamlit/secrets.toml -> HF_API_TOKEN = "hf_xxx"
  2. Or paste it in the sidebar text box at runtime (session only, not stored)
"""

import io
import time
import requests
import streamlit as st
from PIL import Image

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Interior Design Generator",
    page_icon="🏠",
    layout="wide",
)

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
MODEL_OPTIONS = {
    "FLUX.1-dev (best quality, slower)": "black-forest-labs/FLUX.1-dev",
    "Stable Diffusion XL Base 1.0 (balanced)": "stabilityai/stable-diffusion-xl-base-1.0",
    "Stable Diffusion 2.1 (fastest)": "stabilityai/stable-diffusion-2-1",
}

# Rooms pulled from the sample floor plan you provided
ROOM_OPTIONS = {
    "Bedroom (3.0m x 3.0m) - Master": "a bedroom, 3 by 3 meters",
    "Bedroom (3.0m x 3.0m) - Second": "a bedroom, 3 by 3 meters",
    "Kitchen (3.0m x 2.0m)": "a kitchen, 3 by 2 meters",
    "Dining Area (3.0m x 2.3m)": "a dining area, 3 by 2.3 meters",
    "Living Room (3.0m x 3.0m)": "a living room, 3 by 3 meters",
    "Bathroom": "a bathroom",
    "Porch": "a covered front porch",
}

STYLE_OPTIONS = [
    "Modern Minimalist", "Scandinavian", "Industrial", "Contemporary Luxury",
    "Mid-Century Modern", "Japandi", "Rustic Farmhouse", "Bohemian",
    "Traditional", "Coastal",
]

WALL_MATERIALS = [
    "painted drywall", "exposed brick", "natural wood panel", "polished concrete",
    "textured plaster", "marble accent wall", "wallpaper with subtle pattern", "shiplap",
]

FLOOR_MATERIALS = [
    "light oak hardwood flooring", "polished porcelain tile", "matte concrete flooring",
    "large-format marble tile", "natural stone tile", "bamboo flooring",
    "herringbone parquet", "textured area rug over hardwood",
]

WINDOW_OPTIONS = [
    "large floor-to-ceiling glass windows", "black-framed steel windows",
    "wooden-framed windows with sheer curtains", "bay window with cushioned seat",
    "sliding glass doors leading to a small garden", "minimalist aluminum-frame windows",
]

DOOR_OPTIONS = [
    "solid wood panel door", "modern black flush door", "frosted glass sliding door",
    "French double doors", "barn-style sliding door", "minimalist white door",
]

LIGHTING_OPTIONS = [
    "warm ambient lighting", "recessed ceiling lights", "statement pendant lighting",
    "natural daylight streaming in", "soft cove lighting", "industrial track lighting",
]

COLOR_PALETTES = [
    "warm neutral tones (beige, cream, taupe)", "cool monochrome (white, grey, black)",
    "earthy tones (terracotta, olive, sand)", "soft pastel palette",
    "bold contrast (navy and white)", "natural wood and greenery accents",
]

# --------------------------------------------------------------------------
# Sidebar - API key & model
# --------------------------------------------------------------------------
st.sidebar.header("⚙️ Settings")

default_token = st.secrets.get("HF_API_TOKEN", "") if hasattr(st, "secrets") else ""
hf_token = st.sidebar.text_input(
    "Hugging Face API Token",
    value=default_token,
    type="password",
    help="Create a free token at https://huggingface.co/settings/tokens (Read access is enough).",
)

model_label = st.sidebar.selectbox("Image generation model", list(MODEL_OPTIONS.keys()))
model_id = MODEL_OPTIONS[model_label]

with st.sidebar.expander("Advanced generation settings"):
    guidance_scale = st.slider("Guidance scale (prompt adherence)", 1.0, 15.0, 7.5, 0.5)
    num_inference_steps = st.slider("Inference steps (quality vs speed)", 10, 50, 30, 5)
    negative_prompt = st.text_area(
        "Negative prompt",
        value="blurry, low quality, distorted proportions, watermark, text, people, cartoon",
    )

st.sidebar.markdown("---")
st.sidebar.caption(
    "This app calls the Hugging Face Inference API. Some models require you to "
    "accept usage terms on the model's page before your token can use them, and "
    "large models may take ~20-60s to 'warm up' on first request."
)

# --------------------------------------------------------------------------
# Header + reference floor plan
# --------------------------------------------------------------------------
st.title("🏠 AI Interior Design Generator")
st.write(
    "Pick a room from the floor plan, choose your design preferences, and "
    "generate a photorealistic interior concept image."
)

col_plan, col_form = st.columns([1, 1.4], gap="large")

with col_plan:
    st.subheader("Reference Floor Plan")
    uploaded_plan = st.file_uploader(
        "Upload a floor plan (optional - shown for reference only)",
        type=["png", "jpg", "jpeg"],
    )
    if uploaded_plan is not None:
        st.image(uploaded_plan, use_container_width=True, caption="Your floor plan")
    else:
        st.info(
            "No floor plan uploaded. You can still generate a room design below - "
            "room sizes default to the sample layout (two 3x3m bedrooms, "
            "3x2m kitchen, 3x2.3m dining, 3x3m living room, plus bathroom and porch)."
        )

# --------------------------------------------------------------------------
# Design requirement form
# --------------------------------------------------------------------------
with col_form:
    st.subheader("Design Requirements")

    room_label = st.selectbox("Room", list(ROOM_OPTIONS.keys()))
    style = st.selectbox("Interior style", STYLE_OPTIONS)

    c1, c2 = st.columns(2)
    with c1:
        wall_material = st.selectbox("Wall material / finish", WALL_MATERIALS)
        window_choice = st.selectbox("Window style", WINDOW_OPTIONS)
        lighting = st.selectbox("Lighting", LIGHTING_OPTIONS)
    with c2:
        floor_material = st.selectbox("Flooring material", FLOOR_MATERIALS)
        door_choice = st.selectbox("Door style", DOOR_OPTIONS)
        palette = st.selectbox("Color palette", COLOR_PALETTES)

    extra_details = st.text_area(
        "Additional details (furniture, mood, specific requests)",
        placeholder="e.g. add a reading nook, include indoor plants, cozy evening mood",
    )

    aspect_ratio = st.radio("Image orientation", ["Landscape (wide room shot)", "Square"], horizontal=True)
    width, height = (1024, 768) if aspect_ratio.startswith("Landscape") else (1024, 1024)

# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------
def build_prompt() -> str:
    room_desc = ROOM_OPTIONS[room_label]
    parts = [
        f"Photorealistic interior design render of {room_desc}",
        f"in a {style.lower()} style",
        f"with {wall_material} walls",
        f"{floor_material} flooring",
        f"{window_choice}",
        f"a {door_choice}",
        f"{lighting}",
        f"using a {palette} color palette",
    ]
    if extra_details.strip():
        parts.append(extra_details.strip())
    parts.append(
        "architectural digest photography, ultra realistic, high detail, "
        "professional interior photography, wide angle lens, natural shadows, 8k"
    )
    return ", ".join(parts)

prompt = build_prompt()

st.subheader("Generated Prompt")
st.code(prompt, language="text")

# --------------------------------------------------------------------------
# Hugging Face Inference call
# --------------------------------------------------------------------------
def generate_image(api_token: str, model: str, prompt: str, negative_prompt: str,
                    guidance_scale: float, num_inference_steps: int,
                    width: int, height: int, max_retries: int = 3):
    """Call the HF Inference API and return a PIL Image, or raise an Exception."""
    # Hugging Face retired api-inference.huggingface.co in 2025 (it now returns
    # HTTP 410 / fails to resolve on some networks). All serverless inference
    # now goes through the router, routed to the "hf-inference" provider.
    api_url = f"https://router.huggingface.co/hf-inference/models/{model}"
    headers = {"Authorization": f"Bearer {api_token}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "width": width,
            "height": height,
        },
        "options": {"wait_for_model": True},
    }

    for attempt in range(max_retries):
        response = requests.post(api_url, headers=headers, json=payload, timeout=180)

        if response.status_code == 200:
            content_type = response.headers.get("content-type", "")
            if "image" in content_type:
                return Image.open(io.BytesIO(response.content))
            # Some models return JSON with base64 or an error message
            try:
                data = response.json()
            except ValueError:
                raise RuntimeError("Unexpected response format from the API.")
            raise RuntimeError(f"API did not return an image: {data}")

        if response.status_code == 503:
            # Model is loading - wait and retry
            wait_s = 15
            try:
                wait_s = response.json().get("estimated_time", 15)
            except ValueError:
                pass
            st.info(f"Model is warming up, retrying in {int(wait_s)}s... "
                    f"(attempt {attempt + 1}/{max_retries})")
            time.sleep(min(wait_s, 30))
            continue

        if response.status_code == 401:
            raise RuntimeError("Invalid or missing Hugging Face API token.")

        if response.status_code == 403:
            raise RuntimeError(
                "Access denied. This model may require you to accept its license "
                "on its Hugging Face page before your token can use it."
            )

        raise RuntimeError(f"API error {response.status_code}: {response.text[:300]}")

    raise RuntimeError("Model did not finish loading after several retries. Please try again.")


st.markdown("---")
generate_clicked = st.button("✨ Generate Interior Design", type="primary", use_container_width=True)

if "generated_images" not in st.session_state:
    st.session_state.generated_images = []

if generate_clicked:
    if not hf_token:
        st.error("Please enter your Hugging Face API token in the sidebar first.")
    else:
        with st.spinner(f"Generating with {model_label}... this can take up to a minute."):
            try:
                image = generate_image(
                    api_token=hf_token,
                    model=model_id,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    width=width,
                    height=height,
                )
                st.session_state.generated_images.insert(0, {
                    "image": image,
                    "prompt": prompt,
                    "room": room_label,
                    "style": style,
                })
                st.success("Image generated!")
            except RuntimeError as e:
                st.error(str(e))
            except requests.exceptions.Timeout:
                st.error("The request timed out. Try again or pick a faster model.")
            except requests.exceptions.ConnectionError:
                st.error(
                    "Couldn't reach the Hugging Face API. This can happen if your "
                    "network/firewall blocks router.huggingface.co, or Hugging Face "
                    "is having connectivity issues. Try again in a moment."
                )
            except Exception as e:
                st.error(f"Something went wrong: {e}")

# --------------------------------------------------------------------------
# Results gallery
# --------------------------------------------------------------------------
if st.session_state.generated_images:
    st.subheader("Results")
    for idx, item in enumerate(st.session_state.generated_images):
        cols = st.columns([2, 1])
        with cols[0]:
            st.image(item["image"], use_container_width=True,
                      caption=f"{item['room']} — {item['style']}")
        with cols[1]:
            st.caption("Prompt used:")
            st.write(item["prompt"])
            buf = io.BytesIO()
            item["image"].save(buf, format="PNG")
            st.download_button(
                "⬇️ Download image",
                data=buf.getvalue(),
                file_name=f"design_{idx}.png",
                mime="image/png",
                key=f"dl_{idx}",
            )
        st.markdown("---")
