from supertonic import TTS

print("Loading model...")

tts = TTS(auto_download=True)

print("Model loaded")

# 사용 가능한 음성 확인
try:
    voices = tts.list_voice_styles()
    print("Voices:", voices)
except Exception as e:
    print("Voice list error:", e)

# M1 음성 테스트
try:
    style = tts.get_voice_style("M1")
    print("Style:", style)

    result = tts.synthesize(
        "Hello world",
        voice_style=style
    )

    print("Result type:", type(result))
    print("Result:", result)

except Exception as e:
    import traceback
    traceback.print_exc()

wav, duration = tts.synthesize(
    "Hello world",
    voice_style=style
)

print(wav.shape)