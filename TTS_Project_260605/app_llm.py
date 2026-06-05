from flask import (
    Flask,
    render_template,
    request,
    send_file,
    jsonify
)

from supertonic import TTS

import io
import traceback
import numpy as np
import soundfile as sf

# =========================
# LangChain / Local LLM
# =========================

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# =========================
# Flask
# =========================

app = Flask(__name__)


# =========================
# TTS
# =========================

print("Loading Supertonic model...")

tts = TTS(auto_download=True)

print("Supertonic loaded.")

VOICE_LIST = [
    "M1",
    "F1",
    "F2"
]


# =========================
# Local LLM
# =========================

print("Initializing Local LLM...")

llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="lmstudio-community/qwen/qwen3-coder-30b",
    temperature=0.1,
)

prompt = ChatPromptTemplate.from_template(
    """
{input}

Please give me a brief and kindly reply
in 3 lines or less.
"""
)

chain = prompt | llm | StrOutputParser()

print("Local LLM ready.")


# =========================
# Routes
# =========================

@app.route("/")
def index():

    return render_template(
        "index.html",
        voices=VOICE_LIST
    )


# ==================================
# LLM Response API
# ==================================

@app.route("/ask", methods=["POST"])
def ask_ai():

    try:

        data = request.get_json()

        user_text = data.get(
            "text",
            ""
        ).strip()

        if not user_text:

            return jsonify(
                {
                    "error": "No text provided"
                }
            ), 400

        print("\n" + "=" * 60)
        print("USER INPUT")
        print(user_text)

        llm_response = chain.invoke(
            {
                "input": user_text
            }
        )

        print("\nLLM RESPONSE")
        print(llm_response)

        return jsonify(
            {
                "response": llm_response
            }
        )

    except Exception:

        traceback.print_exc()

        return jsonify(
            {
                "error": "LLM processing failed"
            }
        ), 500


# ==================================
# TTS API
# ==================================

@app.route("/speak", methods=["POST"])
def speak():

    try:

        data = request.get_json()

        text = data.get(
            "text",
            ""
        )

        voice_name = data.get(
            "voice",
            "M1"
        )

        if not text:

            return "No text", 400

        print("\nGenerating speech...")
        print("Voice:", voice_name)

        style = tts.get_voice_style(
            voice_name=voice_name
        )

        wav, duration = tts.synthesize(
            text,
            voice_style=style
        )

        wav = np.asarray(
            wav
        ).squeeze()

        print("wav shape:", wav.shape)
        print("duration:", duration)

        buffer = io.BytesIO()

        sf.write(
            buffer,
            wav,
            samplerate=48000,
            format="WAV"
        )

        buffer.seek(0)

        print("Audio generation complete.")

        return send_file(
            buffer,
            mimetype="audio/wav",
            as_attachment=False
        )

    except Exception:

        traceback.print_exc()

        return "TTS generation failed", 500


# ==================================
# Health Check
# ==================================

@app.route("/health")
def health():

    return {
        "status": "ok",
        "tts": "loaded",
        "llm": "loaded"
    }


# ==================================
# Main
# ==================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=8001,
        debug=True
    )