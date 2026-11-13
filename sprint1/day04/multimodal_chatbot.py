"""
Multimodal Chatbot - Streamlit + Hugging Face (100% free, runs locally).

Every model is an open model from the Hugging Face Hub, downloaded once and
run on your own CPU - no Inference Providers, no credits, no billing.

  transformers.pipeline:
    - Text          -> Text   : Qwen/Qwen2.5-0.5B-Instruct           (text-generation)
    - Image + Text  -> Text   : HuggingFaceTB/SmolVLM-256M-Instruct  (image-text-to-text)
    - Image         -> Text   : Salesforce/blip-image-captioning-base (image-to-text)
    - Speech        -> Text   : openai/whisper-tiny                  (automatic-speech-recognition)
    - Text          -> Speech : facebook/mms-tts-eng                 (text-to-speech)
  diffusers:
    - Text          -> Image  : stabilityai/sd-turbo                 (1-step text-to-image)

How to use the chat:
  - Type a question                       -> Qwen answers (streamed)
  - Attach an image (+ optional question) -> BLIP captions it, SmolVLM answers about it
  - Record with the mic in the sidebar    -> Whisper transcribes, Qwen answers
  - Start a message with  /imagine ...    -> sd-turbo generates an image
  - Turn on "Speak replies"               -> MMS-TTS reads the answer aloud

--- Two profiles ---
  local : everything above (needs ~10GB RAM if every feature is used).
  cloud : "lite" set that fits Streamlit Community Cloud's free tier
          (~2.7GB RAM, 2 CPUs) - one SmolVLM model does both text and image
          chat, BLIP captions and sd-turbo image generation are switched off.
          Measured peak ~1.2GB with all cloud models loaded.
  Auto-detected (Community Cloud mounts apps under /mount/src); override with
  APP_PROFILE = "local" | "cloud" in secrets or the environment.

--- One-time setup (local) ---
1. (Optional) HF_TOKEN=hf_xxx in the repo-root .env - only used for faster,
   rate-limit-free model downloads. All models here are public.
2. pip install -r sprint1/day04/requirements.txt
3. streamlit run sprint1/day04/multimodal_chatbot.py

--- Deploy (Streamlit Community Cloud) ---
  Main file: sprint1/day04/multimodal_chatbot.py  (requirements.txt next to it is used)
  Secrets  : HF_TOKEN = "hf_xxx"   (optional, see secrets.toml.example)

Each model downloads the first time its feature is used, then loads from
the local cache. Total ~6GB on disk locally (sd-turbo is ~2.5GB).
Docs: https://huggingface.co/docs/transformers/main_classes/pipelines
      https://huggingface.co/docs/diffusers/using-diffusers/sdxl_turbo
"""
import hashlib
import io
import os
import threading

import numpy as np
import soundfile as sf
import streamlit as st
import torch
from dotenv import load_dotenv
from PIL import Image
from transformers import TextIteratorStreamer, pipeline

load_dotenv()  # local: reads the repo-root .env into the environment


def get_setting(name: str, default: str = "") -> str:
    """Streamlit secrets first (Community Cloud / secrets.toml), then env / .env."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # no secrets.toml anywhere - that's fine locally
        pass
    return os.getenv(name, default)


HF_TOKEN = get_setting("HF_TOKEN") or None
PROFILE = get_setting("APP_PROFILE") or ("cloud" if os.path.isdir("/mount/src") else "local")
IS_CLOUD = PROFILE == "cloud"

# Cloud CPUs are capped at 2 cores even if the host reports more.
torch.set_num_threads(2 if IS_CLOUD else (os.cpu_count() or 4))

if IS_CLOUD:  # one small VLM handles both text and image chat to stay well under 2.7GB
    CHAT_MODELS = VISION_MODELS = ["HuggingFaceTB/SmolVLM-256M-Instruct"]
else:
    CHAT_MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]
    VISION_MODELS = ["HuggingFaceTB/SmolVLM-256M-Instruct", "HuggingFaceTB/SmolVLM-500M-Instruct"]
CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
ASR_MODEL = "openai/whisper-tiny"
TTS_MODEL = "facebook/mms-tts-eng"
IMAGE_GEN_MODEL = "stabilityai/sd-turbo"

SYSTEM_PROMPT = "You are a helpful, concise multimodal assistant. Keep answers short."
MAX_NEW_TOKENS = 256

st.set_page_config(page_title="Multimodal Chatbot", page_icon="🤖")


# ---------- Local models (each loaded once, then cached) ----------

@st.cache_resource(show_spinner="Loading local model (first run downloads it)...")
def load_pipeline(task: str, model: str):
    pipe = pipeline(task, model=model, token=HF_TOKEN)
    if task == "image-text-to-text":
        # SmolVLM upscales and tiles images by default - ~15x slower on CPU.
        ip = pipe.processor.image_processor
        ip.do_image_splitting = False
        ip.size = {"longest_edge": 512}
    return pipe


@st.cache_resource(show_spinner="Loading sd-turbo (first run downloads ~2.5GB)...")
def load_image_generator():
    from diffusers import AutoPipelineForText2Image

    return AutoPipelineForText2Image.from_pretrained(
        IMAGE_GEN_MODEL, variant="fp16", torch_dtype=torch.float32  # fp16 download, fp32 on CPU
    )


def stream_generate(pipe, tokenizer, **kwargs):
    """Run a pipeline in a background thread and yield tokens as they are produced."""
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=300)
    if pipe.task == "text-generation":  # text pipeline takes generate args directly
        kwargs["streamer"] = streamer
    else:  # image-text-to-text wants them wrapped in generate_kwargs
        kwargs.setdefault("generate_kwargs", {})["streamer"] = streamer
    errors = []

    def run():
        try:
            pipe(**kwargs)
        except Exception as e:  # surface thread errors instead of hanging the stream
            errors.append(e)
            streamer.end()

    threading.Thread(target=run, daemon=True).start()
    yield from streamer
    if errors:
        raise errors[0]


def chat_reply(messages: list[dict], model: str):
    """Text -> text with a small instruct LLM (or the VLM, in the cloud profile)."""
    if model in VISION_MODELS:  # a VLM chats too - just send text-only content
        pipe = load_pipeline("image-text-to-text", model)
        vlm_msgs = [
            {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
            for m in messages if m["role"] != "system"
        ]
        return stream_generate(pipe, pipe.processor.tokenizer, text=vlm_msgs, max_new_tokens=MAX_NEW_TOKENS)
    pipe = load_pipeline("text-generation", model)
    return stream_generate(pipe, pipe.tokenizer, text_inputs=messages, max_new_tokens=MAX_NEW_TOKENS)


def vision_reply(image: Image.Image, question: str, model: str):
    """Image + text -> text with a small vision-language model."""
    pipe = load_pipeline("image-text-to-text", model)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": question},
    ]}]
    return stream_generate(pipe, pipe.processor.tokenizer, text=messages, max_new_tokens=MAX_NEW_TOKENS)


def transcribe(wav_bytes: bytes) -> str:
    """Speech -> text with Whisper. Whisper expects 16kHz mono float audio."""
    audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if audio.ndim > 1:  # stereo -> mono
        audio = audio.mean(axis=1)
    if sr != 16000:  # simple linear resample, avoids needing ffmpeg/torchaudio
        n = int(len(audio) * 16000 / sr)
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio)
        audio = audio.astype(np.float32)
    asr = load_pipeline("automatic-speech-recognition", ASR_MODEL)
    return asr({"raw": audio, "sampling_rate": 16000})["text"].strip()


def caption_image(image: Image.Image) -> str:
    """Image -> short caption with BLIP."""
    return load_pipeline("image-to-text", CAPTION_MODEL)(image)[0]["generated_text"]


def speak(text: str) -> bytes:
    """Text -> speech with MMS-TTS. Returns WAV bytes for st.audio."""
    out = load_pipeline("text-to-speech", TTS_MODEL)(text[:600])  # short = fast on CPU
    buf = io.BytesIO()
    sf.write(buf, np.squeeze(out["audio"]), out["sampling_rate"], format="WAV")
    return buf.getvalue()


def generate_image(prompt: str) -> bytes:
    """Text -> image with sd-turbo (a single denoising step)."""
    img = load_image_generator()(prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------- Helpers ----------

def history_for_llm(messages: list[dict]) -> list[dict]:
    """Previous turns as plain text (past images are replaced by their caption)."""
    llm_msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in messages:
        text = m.get("content", "")
        if m.get("caption"):
            text = f"[User shared an image: {m['caption']}] {text}"
        elif m.get("image"):  # cloud profile has no captioner
            text = f"[User shared an image] {text}"
        if text:
            llm_msgs.append({"role": m["role"], "content": text})
    return llm_msgs


def render(m: dict) -> None:
    if m.get("image"):
        st.image(m["image"], width=260)
    if m.get("caption"):
        st.caption(f"🏷️ BLIP caption: *{m['caption']}*")
    if m.get("voice"):
        st.caption("🎙️ Transcribed by Whisper")
    if m.get("content"):
        st.markdown(m["content"])
    if m.get("gen_image"):
        st.image(m["gen_image"])
    if m.get("audio"):
        st.audio(m["audio"], format="audio/wav")


# ---------- Sidebar ----------

with st.sidebar:
    st.header("⚙️ Settings")
    if IS_CLOUD:
        st.info(
            "☁️ **Cloud (lite) mode** — one small model handles text and images to fit the "
            "free tier's memory. Image generation and BLIP captions are available when you "
            "run the app on your own computer.",
        )
    chat_model = st.selectbox("Chat model", CHAT_MODELS, help="Bigger = smarter but slower on CPU.")
    vision_model = st.selectbox("Vision model", VISION_MODELS)
    speak_replies = st.toggle("🔊 Speak replies")
    st.caption("All models are free, open Hugging Face models running on this server's CPU.")

    st.divider()
    voice = st.audio_input("🎙️ Ask by voice")

    st.divider()
    st.markdown(
        "**Try:**\n"
        "- Ask any question\n"
        "- 📎 Attach an image and ask about it\n"
        "- 🎙️ Record a question\n"
        + ("" if IS_CLOUD else "- `/imagine a cat astronaut on the moon`")
    )
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("🤖 Multimodal Chatbot")
st.caption("Text · Image · Voice — free open models from 🤗 Hugging Face, no paid APIs")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        render(msg)


# ---------- Collect input (typed / image / voice) ----------

prompt = st.chat_input(
    "Ask something or attach an image" + ("" if IS_CLOUD else ", or type /imagine ..."),
    accept_file=True,
    file_type=["png", "jpg", "jpeg", "webp"],
)

user_text, image, from_voice = "", None, False
if prompt:
    user_text = (prompt.text or "").strip()
    if prompt.files:
        image = Image.open(prompt.files[0]).convert("RGB")
elif voice is not None:
    wav = voice.getvalue()
    digest = hashlib.md5(wav).hexdigest()
    if digest != st.session_state.get("last_voice"):  # process each recording once
        st.session_state.last_voice = digest
        with st.spinner("Transcribing with Whisper..."):
            user_text = transcribe(wav)
        from_voice = True

if not (user_text or image):
    st.stop()

# ---------- User turn ----------

history = history_for_llm(st.session_state.messages)  # built before adding this turn
user_msg = {"role": "user", "content": user_text, "voice": from_voice}

if image is not None:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    user_msg["image"] = buf.getvalue()
    if not IS_CLOUD:  # BLIP would add ~1GB of memory on the cloud
        with st.spinner("Captioning image with BLIP..."):
            user_msg["caption"] = caption_image(image)
    if not user_text:
        user_msg["content"] = user_text = "Describe this image."

st.session_state.messages.append(user_msg)
with st.chat_message("user"):
    render(user_msg)

# ---------- Assistant turn ----------

reply = {"role": "assistant", "content": ""}
with st.chat_message("assistant"):
    try:
        if user_text.lower().startswith("/imagine") and IS_CLOUD:
            reply["content"] = (
                "🎨 Image generation is turned off in the cloud version: the image model needs "
                "about 5 GB of memory, and the free tier has about 2.7 GB. Run the app on your "
                "own computer to try `/imagine`."
            )
            st.markdown(reply["content"])
        elif user_text.lower().startswith("/imagine"):
            image_prompt = user_text[len("/imagine"):].strip() or "a friendly robot waving hello"
            with st.spinner("Generating image with sd-turbo (~10-40s on CPU)..."):
                reply["gen_image"] = generate_image(image_prompt)
            reply["content"] = f"Here's what I imagined for: *{image_prompt}*"
            st.markdown(reply["content"])
            st.image(reply["gen_image"])
        elif image is not None:
            with st.spinner("Looking at the image..."):
                tokens = vision_reply(image, user_text, vision_model)
            reply["content"] = st.write_stream(tokens).strip()
        else:
            with st.spinner("Thinking..."):
                tokens = chat_reply(history + [{"role": "user", "content": user_text}], chat_model)
            reply["content"] = st.write_stream(tokens).strip()

        if speak_replies and reply["content"] and not reply.get("gen_image"):
            with st.spinner("Synthesizing speech..."):
                reply["audio"] = speak(reply["content"])
            st.audio(reply["audio"], format="audio/wav", autoplay=True)
    except Exception as e:
        reply["content"] = f"⚠️ Error: {e}"
        st.error(reply["content"])

st.session_state.messages.append(reply)
