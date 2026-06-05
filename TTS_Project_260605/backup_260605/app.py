from flask import Flask, render_template, request, send_file
from supertonic import TTS

import io
import traceback
import numpy as np
import soundfile as sf

app = Flask(__name__)

# 서버 시작 시 한 번만 모델 로딩
print("Loading TTS model...")
tts = TTS(auto_download=True)
print("TTS model loaded.")

VOICE_LIST = [
    "M1",
    "F1",
    "F2"
]


@app.route("/")
def index():
    return render_template(
        "index.html",
        voices=VOICE_LIST
    )


@app.route("/tts", methods=["POST"])
def generate_tts():

    text = request.form.get("text", "").strip()
    voice_name = request.form.get("voice", "M1")

    print("=" * 60)
    print("TEXT :", text)
    print("VOICE:", voice_name)

    if not text:
        return "텍스트가 입력되지 않았습니다.", 400

    try:

        # 선택된 음성 스타일
        style = tts.get_voice_style(
            voice_name=voice_name
        )

        print("Style loaded.")

        # 음성 생성
        wav, duration = tts.synthesize(
            text,
            voice_style=style
        )

        print("Synthesis complete.")
        print("Original wav shape:", wav.shape)
        print("Duration:", duration)

        # (1, N) -> (N,)
        wav = np.asarray(wav).squeeze()

        print("Converted wav shape:", wav.shape)
        print("wav dtype:", wav.dtype)

        # 메모리 버퍼 생성
        buffer = io.BytesIO()

        # WAV 파일을 메모리에 저장
        sf.write(
            buffer,
            wav,
            samplerate=24000,
            format="WAV"
        )

        buffer.seek(0)

        print("Audio successfully generated.")

        return send_file(
            buffer,
            mimetype="audio/wav",
            as_attachment=False
        )

    except Exception as e:

        print("\nERROR OCCURRED")
        traceback.print_exc()

        return f"""
TTS Error

{str(e)}
""", 500


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=8000,
        debug=True
    )