# Voxtral-Mini-4B API Reference

## Server Information

**Model**: `mistralai/Voxtral-Mini-4B-Realtime-2602`  
**Host**: `<your-server-ip-or-hostname>`  
**Port**: `8000`  
**Model Size**: ~9 GB (3.4B LLM + 970M Audio Encoder)

---

## Health Check

Verify the server is running:

```bash
# Health endpoint
curl http://<your-server>:8000/health

# List available models
curl http://<your-server>:8000/v1/models
```

---

## Batch STT Endpoint

### HTTP POST: `/v1/audio/transcriptions`

OpenAI-compatible audio transcription endpoint for offline (non-streaming) transcription.

### cURL Example

```bash
curl -X POST http://<your-server>:8000/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/audio.wav" \
  -F "model=mistralai/Voxtral-Mini-4B-Realtime-2602" \
  -F "temperature=0.0"
```

### Python Example (requests)

```python
import requests

url = "http://<your-server>:8000/v1/audio/transcriptions"

with open("audio.wav", "rb") as audio_file:
    files = {"file": audio_file}
    data = {
        "model": "mistralai/Voxtral-Mini-4B-Realtime-2602",
        "temperature": 0.0
    }
    response = requests.post(url, files=files, data=data)
    
print(response.json())
# Expected response: {"text": "transcribed text here"}
```

### Python Example (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    api_key="dummy",  # Not required for vLLM
    base_url="http://<your-server>:8000/v1"
)

with open("audio.wav", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="mistralai/Voxtral-Mini-4B-Realtime-2602",
        file=audio_file,
        temperature=0.0
    )

print(transcript.text)
```

---

## Streaming STT Endpoint

### WebSocket: `ws://<your-server>:8000/v1/realtime`

Real-time streaming transcription via WebSocket connection.

### Python Example (websockets)

```python
import websockets
import asyncio
import json

async def stream_audio(audio_path: str):
    uri = "ws://<your-server>:8000/v1/realtime"
    
    async with websockets.connect(uri) as ws:
        # Send audio in chunks
        with open(audio_path, "rb") as f:
            # Skip WAV header (44 bytes for standard WAV)
            f.read(44)
            
            while chunk := f.read(4096):  # 4KB chunks (~128ms @ 16kHz PCM16)
                await ws.send(chunk)
                
                # Optional: receive intermediate results
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=0.1)
                    result = json.loads(message)
                    print(f"Partial: {result}")
                except asyncio.TimeoutError:
                    pass
        
        # Signal end of audio
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        
        # Receive final transcription
        async for message in ws:
            result = json.loads(message)
            print(f"Final: {result}")
            break

# Run
asyncio.run(stream_audio("audio.wav"))
```

### JavaScript Example (Browser)

```javascript
const ws = new WebSocket('ws://<your-server>:8000/v1/realtime');

ws.onopen = () => {
    console.log('Connected to Voxtral streaming endpoint');
    
    // Send audio chunks from microphone or file
    // Format: PCM16, 16kHz, mono
    const audioChunk = new Uint8Array(4096);
    ws.send(audioChunk);
};

ws.onmessage = (event) => {
    const result = JSON.parse(event.data);
    console.log('Transcription:', result);
};

ws.onerror = (error) => {
    console.error('WebSocket error:', error);
};

ws.onclose = () => {
    console.log('Connection closed');
};
```

---

## Audio Format Requirements

- **Sample Rate**: 16 kHz (recommended)
- **Bit Depth**: 16-bit PCM
- **Channels**: Mono
- **Format**: WAV, FLAC, or raw PCM

### Converting Audio with FFmpeg

```bash
# Convert any audio to Voxtral-compatible format
ffmpeg -i input.mp3 -ar 16000 -ac 1 -sample_fmt s16 output.wav
```

---

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | required | Model identifier |
| `temperature` | float | 0.0 | Sampling temperature (0.0 = greedy, recommended) |
| `file` | binary | required | Audio file (batch endpoint only) |

---

## Performance Characteristics

- **Batch Mode**: Full file transcription, optimized for WER accuracy
- **Streaming Mode**: Real-time transcription with low latency
- **Recommended Chunk Size**: 4096 bytes (~128ms of audio @ 16kHz PCM16)
- **GPU Memory**: ~9 GB with 85% utilization
- **CUDA Graph Mode**: PIECEWISE (optimized for streaming)

---

## Troubleshooting

### Check Server Logs

```bash
docker logs -f vllm-large
```

### Verify GPU Status

```bash
nvidia-smi
```

### Test Network Connectivity

```bash
# TCP connection test
nc -zv <your-server> 8000

# HTTP test
curl -v http://<your-server>:8000/health
```

### Common Issues

1. **Connection refused**: Ensure port 8000 is accessible (check firewall rules)
2. **WebSocket timeout**: Verify streaming endpoint is enabled in vLLM config
3. **Poor WER**: Use temperature=0.0 and ensure audio is 16kHz mono PCM16
4. **Out of memory**: Check GPU memory usage with `nvidia-smi`

---

## Additional Resources

- **Model**: [mistralai/Voxtral-Mini-4B-Realtime-2602](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602)
- **vLLM Docs**: [https://docs.vllm.ai](https://docs.vllm.ai)
- **OpenAI Audio API**: [https://platform.openai.com/docs/api-reference/audio](https://platform.openai.com/docs/api-reference/audio)
