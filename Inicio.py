# app.py
import io
import base64
import os
from typing import Optional

import numpy as np
import pandas as pd  # (por si luego quieres registrar métricas)
from PIL import Image, ImageOps

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from openai import OpenAI

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG BÁSICA
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="🧠 Tablero Inteligente", page_icon="🎨", layout="wide")

# Estado inicial
default_states = dict(
    analysis_done=False,
    full_response="",
    base64_image="",
    last_png_bytes=None,
    story_text="",
)
for k, v in default_states.items():
    st.session_state.setdefault(k, v)

# ──────────────────────────────────────────────────────────────────────────────
# UTILIDADES
# ──────────────────────────────────────────────────────────────────────────────
def get_client(provided_key: Optional[str] = None) -> Optional[OpenAI]:
    """
    Preferimos st.secrets["OPENAI_API_KEY"] si existe.
    Si no hay, usamos la proporcionada por input.
    """
    api_key = None
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    elif provided_key:
        api_key = provided_key.strip()

    if not api_key:
        return None
    # No seteamos variables de entorno globales; mantenemos aislado
    return OpenAI(api_key=api_key)

def npimage_to_png_bytes(img_np: np.ndarray) -> bytes:
    """Convierte un array RGBA/uint8 a PNG en memoria."""
    im = Image.fromarray(img_np.astype("uint8"))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()

def png_bytes_to_b64(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode("utf-8")

def normalize_canvas_rgba(img_np: np.ndarray, to_grayscale: bool = False) -> np.ndarray:
    """
    - Asegura fondo blanco (compone alpha sobre blanco).
    - Opcionalmente pasa a escala de grises (ayuda a modelos a enfocarse en formas).
    - Devuelve RGBA coherente.
    """
    if img_np.dtype != np.uint8:
        img_np = img_np.astype("uint8")

    if img_np.shape[2] == 4:
        rgb = img_np[..., :3]
        alpha = img_np[..., 3:] / 255.0
        white_bg = np.ones_like(rgb, dtype=np.float32) * 255.0
        composed = rgb * alpha + white_bg * (1.0 - alpha)
        rgb = composed.astype("uint8")
    else:
        rgb = img_np[..., :3]

    if to_grayscale:
        pil = Image.fromarray(rgb).convert("L").convert("RGBA")
    else:
        pil = Image.fromarray(rgb).convert("RGBA")

    return np.array(pil)

def is_canvas_blank(img_np: np.ndarray, threshold: float = 0.999) -> bool:
    """
    Consideramos el lienzo "vacío" si casi todo es blanco.
    threshold = porcentaje mínimo de píxeles ~blancos.
    """
    if img_np.shape[2] == 4:
        rgb = img_np[..., :3]
    else:
        rgb = img_np
    # píxeles casi blancos (>= 250 en los 3 canales)
    white_mask = (rgb >= 250).all(axis=2)
    white_ratio = white_mask.mean()
    return white_ratio >= threshold

def stream_chat_completion_image(client: OpenAI, model: str, prompt: str, b64_png: str):
    """
    Hace streaming del análisis de imagen y rinde texto incremental.
    Devuelve el texto completo.
    """
    placeholder = st.empty()
    full_text = ""

    # SDK nuevo: chat.completions con stream=True
    with client.chat.completions.create(
        model=model,
        stream=True,
        messages=[
            {
                "role": "system",
                "content": "Eres un analista visual conciso y claro. Responde en español neutral.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_png}"},
                    },
                ],
            },
        ],
        max_tokens=500,
        temperature=0.2,
    ) as stream:
        for event in stream:
            delta = getattr(getattr(event, "choices", [{}])[0], "delta", None)
            if delta and getattr(delta, "content", None):
                chunk = delta.content
                full_text += chunk
                placeholder.markdown(full_text + "▌")
        placeholder.markdown(full_text)

    return full_text

def chat_simple(client: OpenAI, model: str, prompt: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "Eres creativo y claro."},
                  {"role": "user", "content": prompt}],
        max_tokens=700,
        temperature=0.7,
    )
    return resp.choices[0].message.content or ""

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ℹ️ Acerca de")
    st.caption("Esta app interpreta tu **boceto** y puede generar una **historia infantil** basada en su descripción.")
    st.markdown("---")

    st.markdown("#### 🎛️ Controles del lienzo")
    stroke_width = st.slider("Ancho de línea", 1, 30, 6)
    stroke_color = st.color_picker("Color del trazo", "#000000")
    bg_color = st.color_picker("Color de fondo", "#FFFFFF")
    to_grayscale = st.toggle("Forzar escala de grises (recomendado)", value=True)

    st.markdown("---")
    st.markdown("#### 🔐 Clave de API")
    st.caption("Usa **st.secrets['OPENAI_API_KEY']** en despliegue. Localmente, ingrésala aquí.")
    api_key_input = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")

    model = st.selectbox(
        "Modelo",
        options=["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"],
        index=0,
        help="Modelos multimodales; 'mini' es más económico.",
    )

    st.markdown("---")
    st.markdown("#### 💾 Exportar")
    auto_download = st.toggle("Mostrar botón de descarga del PNG", value=True)

# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────
col_l, col_r = st.columns([1, 2], vertical_alignment="center")
with col_l:
    st.title("🧠 Tablero Inteligente")
    st.markdown(
        "Dibuja en el lienzo y luego **Analiza la imagen**. "
        "Después podrás generar una **historia infantil** basada en la descripción."
    )
with col_r:
    st.markdown(
        """
        <div style="display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end">
          <span style="background:#EEF6FF;border:1px solid #BBD6FF;padding:4px 8px;border-radius:999px">Streamlit</span>
          <span style="background:#EEFFEF;border:1px solid #BBEFC0;padding:4px 8px;border-radius:999px">OpenAI</span>
          <span style="background:#FFF7E6;border:1px solid #FFE0A6;padding:4px 8px;border-radius:999px">Canvas</span>
          <span style="background:#F5E6FF;border:1px solid #E0C6FF;padding:4px 8px;border-radius:999px">Multimodal</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# CANVAS
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("🎨 Lienzo")
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.0)",       # sin relleno por defecto
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=360,
    width=520,
    drawing_mode="freedraw",
    key="canvas",
)

# Botones principales
colA, colB, colC = st.columns([1, 1, 2])
analyze = colA.button("🔎 Analizar imagen", type="primary")
clear = colB.button("🧹 Limpiar lienzo")

if clear:
    # Reinicia estado y fuerza nueva clave del widget canvas con un truco de rerun
    st.session_state.update(
        analysis_done=False, full_response="", base64_image="", last_png_bytes=None, story_text=""
    )
    st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# LÓGICA DE ANÁLISIS
# ──────────────────────────────────────────────────────────────────────────────
if analyze:
    client = get_client(api_key_input)
    if client is None:
        st.warning("🔑 Ingresa tu API key o configura **st.secrets['OPENAI_API_KEY']**.")
    elif canvas_result.image_data is None:
        st.info("✍️ Dibuja algo en el lienzo antes de analizar.")
    else:
        # Normalizamos la imagen
        raw_np = np.array(canvas_result.image_data)
        norm_np = normalize_canvas_rgba(raw_np, to_grayscale=to_grayscale)

        if is_canvas_blank(norm_np):
            st.warning("El lienzo parece vacío (casi todo blanco). Agrega algunos trazos e inténtalo de nuevo.")
        else:
            # Guardamos PNG en memoria y en sesión
            png_bytes = npimage_to_png_bytes(norm_np)
            b64_img = png_bytes_to_b64(png_bytes)
            st.session_state.last_png_bytes = png_bytes
            st.session_state.base64_image = b64_img

            with st.spinner("Analizando boceto con IA…"):
                prompt_text = (
                    "Describe brevemente el boceto. Menciona objetos, relaciones espaciales, "
                    "composición (centro, balance), y la posible intención del autor. Sé específico y claro."
                )
                try:
                    full = stream_chat_completion_image(client, model, prompt_text, b64_img)
                    st.session_state.full_response = full
                    st.session_state.analysis_done = True

                except Exception as e:
                    st.error(f"❌ Ocurrió un error durante el análisis: {e}")
                    st.stop()

# Mostrar resultado del análisis
if st.session_state.analysis_done and st.session_state.full_response:
    st.success("✅ Análisis completado")
    st.markdown("**Descripción generada:**")
    st.write(st.session_state.full_response)

    # Descargar PNG
    if st.session_state.last_png_bytes and auto_download:
        st.download_button(
            "📥 Descargar boceto (PNG)",
            data=st.session_state.last_png_bytes,
            file_name="boceto.png",
            mime="image/png",
        )

# ──────────────────────────────────────────────────────────────────────────────
# HISTORIA INFANTIL
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📚 Generar historia infantil")

col1, col2, col3, col4 = st.columns(4)
edad = col1.selectbox("Edad objetivo", ["3-5", "6-8", "9-11"], index=1)
tono = col2.selectbox("Tono", ["tierno", "aventura", "misterio amable", "fantasía"], index=0)
idioma = col3.selectbox("Idioma", ["español", "inglés"], index=0)
longitud = col4.select_slider("Longitud", options=["muy corta", "corta", "media"], value="corta")
moraleja = st.text_input("Moraleja (opcional)", placeholder="p. ej., La amistad vale más que el oro")
personajes = st.text_input("Personajes (opcional)", placeholder="p. ej., Susi la estrella y Bruno el pez")

gen_story = st.button("✨ Crear historia infantil")

if gen_story:
    if not st.session_state.analysis_done or not st.session_state.full_response:
        st.info("Primero realiza el **análisis del boceto** para usarlo como base de la historia.")
    else:
        client = get_client(api_key_input)
        if client is None:
            st.warning("🔑 Ingresa tu API key o configura **st.secrets['OPENAI_API_KEY']**.")
        else:
            with st.spinner("Escribiendo historia…"):
                target_words = {"muy corta": 120, "corta": 220, "media": 380}[longitud]
                prompt_story = f"""
Eres un escritor/a de literatura infantil.
Crea una historia **original** basada en esta descripción visual del boceto:

DESCRIPCIÓN BASE:
\"\"\"{st.session_state.full_response}\"\"\"

Requisitos:
- Edad objetivo: {edad}.
- Tono: {tono}.
- Idioma: {idioma}.
- Extensión aproximada: {target_words} palabras.
- Personajes a incluir (si se proporcionan): {personajes or "libre"}.
- Incluye una **moraleja** {("explícita: " + moraleja) if moraleja else "implícita y positiva"}.
- Usa lenguaje claro, imágenes mentales ricas y párrafos cortos.
"""
                try:
                    text = chat_simple(client, model, prompt_story)
                    st.session_state.story_text = text
                    st.markdown("**📖 Tu historia:**")
                    st.write(text)
                except Exception as e:
                    st.error(f"❌ Ocurrió un error creando la historia: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# PISTA / CONSEJOS
# ──────────────────────────────────────────────────────────────────────────────
with st.expander("💡 Consejos para mejores resultados"):
    st.markdown(
        """
- Usa **trazos oscuros** y contraste alto.
- Activa **escala de grises** para que el modelo se concentre en las formas.
- Procura una **composición clara** (centro, margen, proporciones).
- Si el análisis queda muy general, añade más **detalles clave** (ojos, puertas, flechas, rótulos).
        """
    )
