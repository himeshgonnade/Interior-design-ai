"""
AI Interior Design Generator
-----------------------------
A Streamlit app that lets a user pick a room from a house floor plan,
choose design parameters (style, material, flooring, windows/doors,
color palette, lighting) and generates a photorealistic interior
render by calling a free, public Hugging Face Space (Gradio app)
running FLUX.1-schnell on shared ZeroGPU hardware.

Why a Space instead of the Inference Providers API? Free Hugging Face
accounts only get $0.10/month of Inference Provider credit (enough for
a couple of paid-provider images before you're cut off). Hugging Face
Spaces on ZeroGPU are a genuinely free, no-credit-card, no-billing way
to run image generation - the trade-off is a shared queue, so it can be
slower and occasionally busy.

Deploy on Streamlit Community Cloud or run locally:
    streamlit run app.py

A Hugging Face token is optional here (it only raises your priority in
the shared queue). If you want to set one, either:
  1. In Streamlit secrets:  .streamlit/secrets.toml -> HF_API_TOKEN = "hf_xxx"
  2. Or paste it in the sidebar text box at runtime (session only, not stored)
"""

import io
import streamlit as st
from PIL import Image
from gradio_client import Client

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
# Official free FLUX.1-schnell demo Space. Apache-2.0 licensed, ungated,
# runs on shared ZeroGPU hardware at no cost. "schnell" (German for "fast")
# is distilled for 1-4 step generation, so it's quick even on shared queue.
SPACE_ID = "black-forest-labs/FLUX.1-schnell"
SPACE_API_NAME = "/infer"

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
# Sidebar - optional token & generation settings
# --------------------------------------------------------------------------
st.sidebar.header("⚙️ Settings")

default_token = st.secrets.get("HF_API_TOKEN", "") if hasattr(st, "secrets") else ""
hf_token = st.sidebar.text_input(
    "Hugging Face token (optional)",
    value=default_token,
    type="password",
    help=(
        "Not required to generate images. Adding a free token from "
        "https://huggingface.co/settings/tokens may improve your position "
        "in the shared ZeroGPU queue."
    ),
)

st.sidebar.caption(
    "This app generates images for free via the public FLUX.1-schnell "
    "Hugging Face Space, running on shared ZeroGPU hardware. No billing, "
    "no credits - but it's a shared queue, so generation can occasionally "
    "be slow or briefly unavailable if the Space is busy or restarting."
)

with st.sidebar.expander("Advanced generation settings"):
    num_inference_steps = st.slider(
        "Inference steps", 1, 8, 4,
        help="FLUX.1-schnell is distilled for ~4 steps. More steps rarely helps and just slows things down.",
    )
    randomize_seed = st.checkbox("Randomize seed each time", value=True)
    seed = st.number_input("Seed (used only if randomize is off)", min_value=0, value=42, step=1)

# --------------------------------------------------------------------------
# Header + reference floor plan
# --------------------------------------------------------------------------
st.title("🏠 AI Interior Design Generator")
st.write(
    "Pick a room from the floor plan, choose your design preferences, and "
    "generate a photorealistic interior concept image - free, via a public "
    "Hugging Face Space."
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
    # FLUX.1-schnell expects width/height as multiples of 32.
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
# Free Hugging Face Space call (gradio_client)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_space_client(token: str | None):
    # Caching the Client avoids re-fetching the Space's API description on
    # every generation, which is slow. Cache key includes the token so
    # switching between "no token" and "with token" gets a fresh client.
    return Client(SPACE_ID, token=token or None)


def generate_image(prompt: str, num_inference_steps: int, width: int, height: int,
                    randomize_seed: bool, seed: int, token: str):
    """Call the free FLUX.1-schnell Space and return (PIL.Image, seed_used)."""
    try:
        client = get_space_client(token)
        result = client.predict(
            prompt=prompt,
            seed=int(seed),
            randomize_seed=randomize_seed,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            api_name=SPACE_API_NAME,
        )
    except Exception as e:
        msg = str(e).lower()
        if "quota" in msg or ("gpu" in msg and "exceeded" in msg):
            raise RuntimeError(
                "The free shared GPU quota for this Space is temporarily "
                "exhausted. Wait a minute and try again, or add a Hugging "
                "Face token in the sidebar for better queue priority."
            ) from e
        if "timeout" in msg or "timed out" in msg:
            raise RuntimeError(
                "The Space took too long to respond (it may be waking up "
                "from sleep). Please try again."
            ) from e
        raise RuntimeError(
            f"Couldn't reach the free generation Space right now: {e}"
        ) from e

    # The Space returns (image, seed_used); the image element is typically a
    # local filepath (or a dict with a "path"/"url" key) depending on gradio
    # version, so handle both.
    image_result, used_seed = result[0], result[1]
    if isinstance(image_result, dict):
        image_path = image_result.get("path") or image_result.get("url")
    else:
        image_path = image_result

    image = Image.open(image_path)
    return image, used_seed


st.markdown("---")
generate_clicked = st.button("✨ Generate Interior Design", type="primary", use_container_width=True)

if "generated_images" not in st.session_state:
    st.session_state.generated_images = []

if generate_clicked:
    with st.spinner("Generating via the free FLUX.1-schnell Space... this can take 10-60s depending on the queue."):
        try:
            image, used_seed = generate_image(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                width=width,
                height=height,
                randomize_seed=randomize_seed,
                seed=seed,
                token=hf_token,
            )
            st.session_state.generated_images.insert(0, {
                "image": image,
                "prompt": prompt,
                "room": room_label,
                "style": style,
                "seed": used_seed,
            })
            st.success("Image generated!")
        except RuntimeError as e:
            st.error(str(e))
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
            st.caption(f"Seed: {item['seed']}")
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
