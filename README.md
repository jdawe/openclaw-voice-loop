# OpenClaw Voice Loop 🎙️

Talk to your AI agent like a phone call.

**Mic → Whisper → OpenClaw → TTS → Speaker**

A continuous voice conversation loop that captures your speech, transcribes it locally with Whisper, sends it to any OpenClaw gateway (any model), and speaks the response back to you.

## How It Works

1. **Listen** — Records from your mic, detects speech, stops on silence
2. **Transcribe** — Local Whisper model converts speech to text
3. **Think** — Sends text to your OpenClaw agent via `openclaw agent` CLI
4. **Speak** — Converts the reply to audio via ElevenLabs TTS (or macOS `say` as fallback)
5. **Repeat**

Works with any OpenClaw gateway and any model configured on that gateway (Claude, GPT, Gemini, local models, etc.).

## Requirements

- Python 3.9+
- [OpenClaw](https://openclaw.com) CLI installed and configured
- ffmpeg and portaudio

### macOS

```bash
brew install ffmpeg portaudio
```

### Linux

```bash
sudo apt install ffmpeg portaudio19-dev
```

## Setup

```bash
git clone https://github.com/jdawe/openclaw-voice-loop.git
cd openclaw-voice-loop
pip install -r requirements.txt
```

## Usage

```bash
# Basic — uses macOS `say` for TTS
python voice_loop.py

# With ElevenLabs TTS
export ELEVENLABS_API_KEY=your_key_here
python voice_loop.py

# With a remote OpenClaw gateway
export OPENCLAW_GATEWAY_URL=wss://your-gateway.example.com
export OPENCLAW_GATEWAY_TOKEN=your_token
python voice_loop.py
```

## Configuration

All configuration is via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ELEVENLABS_API_KEY` | _(none)_ | ElevenLabs API key. If unset, falls back to macOS `say` |
| `ELEVENLABS_VOICE_ID` | `21m00Tcm4TlvDq8ikWAM` | ElevenLabs voice ID (default: Rachel) |
| `ELEVENLABS_SPEED` | `1.0` | Playback speed multiplier |
| `WHISPER_MODEL` | `tiny` | Whisper model: `tiny`, `base`, `small`, `medium`, `large` |
| `OPENCLAW_GATEWAY_URL` | _(none)_ | Remote gateway WebSocket URL |
| `OPENCLAW_GATEWAY_TOKEN` | _(none)_ | Gateway auth token |
| `VOICE_SESSION_ID` | `voice-loop` | OpenClaw session ID (maintains conversation context) |
| `AGENT_TIMEOUT` | `60` | Seconds to wait for agent reply |

## Tips

- **First run** downloads the Whisper model (~75MB for `tiny`). Subsequent runs are instant.
- Use `tiny` or `base` Whisper models for fastest transcription. `small` is a good accuracy/speed tradeoff.
- The loop auto-calibrates your mic on startup — stay quiet for 1 second.
- Whisper hallucination filtering is built in (ignores phantom "thank you" / "bye" transcriptions).
- Sessions persist across turns, so the agent remembers context within a conversation.

## License

MIT
