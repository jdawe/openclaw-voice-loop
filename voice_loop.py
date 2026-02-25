#!/usr/bin/env python3
"""Voice conversation loop for OpenClaw.

Mic → Whisper → OpenClaw Gateway → TTS → Speaker

Talk to your AI agent like a phone call.

Configuration (environment variables):
  OPENCLAW_GATEWAY_URL    - Gateway WebSocket URL (optional, for remote gateways)
  OPENCLAW_GATEWAY_TOKEN  - Gateway auth token (optional)
  ELEVENLABS_API_KEY      - ElevenLabs API key (optional, highest TTS priority)
  ELEVENLABS_VOICE_ID     - ElevenLabs voice ID (default: Rachel)
  ELEVENLABS_SPEED        - Playback speed multiplier (default: 1.0)
  OPENAI_API_KEY          - OpenAI API key (optional, second TTS priority)
  OPENAI_VOICE            - OpenAI TTS voice (default: alloy)
  WHISPER_MODEL           - Whisper model size (default: tiny)
  VOICE_SESSION_ID        - OpenClaw session ID (default: voice-loop)
  AGENT_TIMEOUT           - Seconds to wait for agent reply (default: 60)
  SAY_RATE                - macOS `say` words per minute (default: 350)
  MAX_TURNS               - Max conversation turns before reset (default: 50)

Requirements:
  pip install -r requirements.txt
  brew install ffmpeg portaudio   # macOS
  # or: apt install ffmpeg portaudio19-dev  # Linux
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import subprocess, sys, json, time, tempfile, wave
import numpy as np
import sounddevice as sd

# ── Config (all from env vars) ──────────────────────────────────────
GATEWAY_URL = os.environ.get("OPENCLAW_GATEWAY_URL", "")
GATEWAY_TOKEN = os.environ.get("OPENCLAW_GATEWAY_TOKEN", "")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel
ELEVENLABS_SPEED = float(os.environ.get("ELEVENLABS_SPEED", "1.0"))
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_VOICE = os.environ.get("OPENAI_VOICE", "alloy")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "tiny")
SESSION_ID = os.environ.get("VOICE_SESSION_ID", "voice-loop")
AGENT_TIMEOUT = int(os.environ.get("AGENT_TIMEOUT", "60"))

SAMPLE_RATE = 16000
CHANNELS = 1
SILENCE_DURATION = 1.5
MIN_SPEECH_DURATION = 0.5
MAX_REPLY_CHARS = 500
MAX_TURNS = int(os.environ.get("MAX_TURNS", "50"))

VOICE_HINT = (
    "[VOICE MODE] You are in a live voice conversation. "
    "The caller handles TTS playback. RULES: "
    "1) Reply with 1-3 SHORT spoken sentences as plain text. "
    "2) No markdown, no bullets, no code, no lists. "
    "3) Do NOT use the tts tool — the caller handles audio. "
    "4) Do NOT use tools unless absolutely necessary. "
    "5) ALWAYS produce a text reply. "
    "User said: "
)

# ── Globals ─────────────────────────────────────────────────────────
whisper_model = None
turn_count = 0
consecutive_errors = 0


def calibrate_mic(duration=1.0):
    """Record silence to set noise threshold."""
    print("🎤 Calibrating mic (stay quiet)...", end=" ", flush=True)
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=CHANNELS, dtype="float32")
    sd.wait()
    rms = np.sqrt(np.mean(audio ** 2))
    threshold = rms * 3.0
    print(f"done (threshold={threshold:.5f})")
    return max(threshold, 0.005)


def record_utterance(threshold):
    """Record until silence detected after speech."""
    print("🎙️  Listening...", end=" ", flush=True)

    audio_chunks = []
    speech_started = False
    silence_start = None
    chunk_size = int(SAMPLE_RATE * 0.1)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32") as stream:
        while True:
            data, _ = stream.read(chunk_size)
            rms = np.sqrt(np.mean(data ** 2))

            if rms > threshold:
                if not speech_started:
                    speech_started = True
                    print("speaking...", end=" ", flush=True)
                silence_start = None
            elif speech_started:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > SILENCE_DURATION:
                    break

            if speech_started:
                audio_chunks.append(data.copy())

    audio = np.concatenate(audio_chunks)
    duration = len(audio) / SAMPLE_RATE
    print(f"got {duration:.1f}s")

    if duration < MIN_SPEECH_DURATION:
        return None
    return audio


def transcribe(audio):
    """Whisper transcription."""
    global whisper_model
    if whisper_model is None:
        import whisper
        print("📦 Loading Whisper model...", end=" ", flush=True)
        whisper_model = whisper.load_model(WHISPER_MODEL)
        print("done")

    t0 = time.time()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
        audio_int16 = (audio * 32767).astype(np.int16)
        with wave.open(f, "wb") as w:
            w.setnchannels(CHANNELS)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(audio_int16.tobytes())

    result = whisper_model.transcribe(tmp, language="en", fp16=False)
    os.unlink(tmp)
    text = result["text"].strip()
    dt = time.time() - t0
    print(f'📝 [{dt:.1f}s] "{text}"')
    return text


def ask_agent(text):
    """Send to OpenClaw agent, get response."""
    global turn_count, consecutive_errors
    t0 = time.time()
    print("🧠 Thinking...", end=" ", flush=True)

    message = VOICE_HINT + text

    try:
        env = os.environ.copy()
        if GATEWAY_URL:
            env["OPENCLAW_GATEWAY_URL"] = GATEWAY_URL
        if GATEWAY_TOKEN:
            env["OPENCLAW_GATEWAY_TOKEN"] = GATEWAY_TOKEN

        result = subprocess.run(
            [
                "openclaw", "agent", "-m", message,
                "--session-id", SESSION_ID,
                "--thinking", "low",
                "--json", "--timeout", str(AGENT_TIMEOUT),
            ],
            capture_output=True, text=True, env=env,
            timeout=AGENT_TIMEOUT + 10,
        )
    except subprocess.TimeoutExpired:
        print("timeout!")
        consecutive_errors += 1
        return "Sorry, that took too long. Try again."

    dt = time.time() - t0

    if result.returncode != 0:
        err = result.stderr[:300] if result.stderr else "unknown error"
        print(f"error ({dt:.1f}s): {err}")
        consecutive_errors += 1
        return "Sorry, I hit an error. Try again."

    try:
        data = json.loads(result.stdout)
        payloads = data.get("result", {}).get("payloads", [])
        reply_parts = [p.get("text", "") for p in payloads if p.get("text")]
        reply = " ".join(reply_parts).strip()

        if not reply:
            print(f"empty reply ({dt:.1f}s)")
            consecutive_errors += 1
            return "I processed that but had nothing to say."

    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"parse error ({dt:.1f}s): {e}")
        consecutive_errors += 1
        return "Sorry, something went wrong parsing the response."

    # Truncate for TTS
    if len(reply) > MAX_REPLY_CHARS:
        truncated = reply[:MAX_REPLY_CHARS]
        last_period = truncated.rfind(".")
        if last_period > MAX_REPLY_CHARS // 2:
            reply = truncated[: last_period + 1]
        else:
            reply = truncated + "..."

    # Strip markdown artifacts
    for ch in ["**", "```", "`", "- ", "* "]:
        reply = reply.replace(ch, "")

    turn_count += 1
    consecutive_errors = 0
    display = reply[:120] + "..." if len(reply) > 120 else reply
    print(f"[{dt:.1f}s] {display}")
    return reply


def speak_elevenlabs(text):
    """ElevenLabs TTS → optional speed adjustment → play."""
    raw_path = tempfile.mktemp(suffix=".mp3")
    fast_path = tempfile.mktemp(suffix=".mp3")

    try:
        subprocess.run(
            [
                "curl", "-s", "-X", "POST",
                f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}",
                "-H", f"xi-api-key: {ELEVENLABS_API_KEY}",
                "-H", "Content-Type: application/json",
                "-d", json.dumps({
                    "text": text,
                    "model_id": "eleven_turbo_v2_5",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                        "style": 0.0,
                        "use_speaker_boost": True,
                    },
                }),
                "-o", raw_path,
            ],
            capture_output=True, text=True, timeout=30,
        )

        if not os.path.exists(raw_path) or os.path.getsize(raw_path) < 1000:
            print("ElevenLabs TTS failed, falling back to macOS say")
            speak_macos(text)
            return

        play_path = raw_path
        if ELEVENLABS_SPEED != 1.0:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", raw_path,
                    "-filter:a", f"atempo={ELEVENLABS_SPEED}",
                    "-q:a", "2", fast_path,
                ],
                capture_output=True, timeout=15,
            )
            if os.path.exists(fast_path) and os.path.getsize(fast_path) > 500:
                play_path = fast_path

        subprocess.run(["afplay", play_path], timeout=60)

    except Exception as e:
        print(f"ElevenLabs error: {e}, falling back to macOS say")
        speak_macos(text)
    finally:
        for p in [raw_path, fast_path]:
            try:
                os.unlink(p)
            except OSError:
                pass


def speak_openai(text):
    """OpenAI TTS → play."""
    raw_path = tempfile.mktemp(suffix=".mp3")
    try:
        subprocess.run(
            [
                "curl", "-s", "-X", "POST",
                "https://api.openai.com/v1/audio/speech",
                "-H", f"Authorization: Bearer {OPENAI_API_KEY}",
                "-H", "Content-Type: application/json",
                "-d", json.dumps({
                    "model": "tts-1",
                    "voice": OPENAI_VOICE,
                    "input": text,
                }),
                "-o", raw_path,
            ],
            capture_output=True, text=True, timeout=30,
        )

        if not os.path.exists(raw_path) or os.path.getsize(raw_path) < 1000:
            print("OpenAI TTS failed, falling back to macOS say")
            speak_macos(text)
            return

        subprocess.run(["afplay", raw_path], timeout=60)

    except Exception as e:
        print(f"OpenAI TTS error: {e}, falling back to macOS say")
        speak_macos(text)
    finally:
        try:
            os.unlink(raw_path)
        except OSError:
            pass


SAY_RATE = int(os.environ.get("SAY_RATE", "350"))  # words per minute (default ~200, 350 = ~1.75x)

def speak_macos(text):
    """Fallback TTS using macOS `say` command."""
    try:
        subprocess.run(["say", "-r", str(SAY_RATE), text], timeout=60)
    except FileNotFoundError:
        print("⚠️  No TTS available (macOS `say` not found)")
    except Exception as e:
        print(f"macOS say error: {e}")


def speak(text):
    """Route to available TTS."""
    t0 = time.time()
    print("🔊 Speaking...", end=" ", flush=True)

    if ELEVENLABS_API_KEY:
        speak_elevenlabs(text)
    elif OPENAI_API_KEY:
        speak_openai(text)
    else:
        speak_macos(text)

    print(f"done ({time.time() - t0:.1f}s)")


def main():
    global turn_count, consecutive_errors

    print("=" * 50)
    print("🎙️  OpenClaw Voice Loop")
    print("=" * 50)
    print(f"Session: {SESSION_ID}")
    print(f"Whisper: {WHISPER_MODEL}")
    tts_name = "ElevenLabs" if ELEVENLABS_API_KEY else "OpenAI" if OPENAI_API_KEY else "macOS say"
    print(f"TTS: {tts_name}")
    if ELEVENLABS_API_KEY and ELEVENLABS_SPEED != 1.0:
        print(f"Speed: {ELEVENLABS_SPEED}x")
    print("Press Ctrl+C to quit\n")

    threshold = calibrate_mic()

    # Prime whisper
    print("📦 Priming Whisper...", end=" ", flush=True)
    transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))
    print("")

    print("\n🟢 Ready! Start talking.\n")

    while True:
        try:
            if turn_count >= MAX_TURNS:
                print(f"\n⚠️  {MAX_TURNS} turns reached, resetting")
                turn_count = 0
                consecutive_errors = 0
            if consecutive_errors >= 3:
                print(f"\n⚠️  {consecutive_errors} consecutive errors, resetting")
                turn_count = 0
                consecutive_errors = 0

            audio = record_utterance(threshold)
            if audio is None:
                print("(too short, ignoring)")
                continue

            text = transcribe(audio)
            if not text or text.lower().strip() in [
                "", "you", "thank you.", "thanks for watching!",
                "thanks for watching.", "thank you for watching.",
                "bye.", "bye", "the end.", "hmm.",
            ]:
                print("(empty/hallucination, ignoring)")
                continue

            t_total = time.time()
            reply = ask_agent(text)
            speak(reply)
            total = time.time() - t_total
            print(f"⏱️  Total turn: {total:.1f}s (turn {turn_count}/{MAX_TURNS})\n")

        except KeyboardInterrupt:
            print("\n\n👋 Bye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            consecutive_errors += 1
            continue


if __name__ == "__main__":
    main()
