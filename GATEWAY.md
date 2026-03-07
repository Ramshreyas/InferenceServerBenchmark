# Inference Server Gateway — K8s Integration

## Backend Details

| Field | Value |
|---|---|
| **Host** | `ecodev-ai-inference-02` (Teleport node) |
| **Public IP** | `78.46.219.175` |
| **Port** | `8000` |
| **API** | OpenAI-compatible (`/v1`) |
| **Model** | `Sehyo/Qwen3.5-122B-A10B-NVFP4` (122B MoE, 10B active) |
| **Auth** | None (add firewall rules or `--api-key` on vLLM side) |
| **Hoster** | Hetzner Cloud FSN1 |
| **Docker restart policy** | `unless-stopped` — auto-recovers from crashes/reboots |

## Endpoints

```
GET  http://78.46.219.175:8000/health              → healthcheck
GET  http://78.46.219.175:8000/v1/models           → list loaded model
POST http://78.46.219.175:8000/v1/chat/completions  → chat (streaming supported)
POST http://78.46.219.175:8000/v1/completions       → text completion
```

## Client Usage (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://inference-backend:8000/v1",  # K8s service name
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="Sehyo/Qwen3.5-122B-A10B-NVFP4",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True
)
```

## K8s Service — Option A: ExternalName

Use if the server is DNS-resolvable from the cluster:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: inference-backend
  namespace: <your-namespace>
spec:
  type: ExternalName
  externalName: ecodev-ai-inference-02  # or 78.46.219.175 if no DNS
```

## K8s Service — Option B: Endpoints (recommended)

Use when pointing at a raw IP:

```yaml
apiVersion: v1
kind: Endpoints
metadata:
  name: inference-backend
  namespace: <your-namespace>
subsets:
  - addresses:
      - ip: 78.46.219.175
    ports:
      - port: 8000
        protocol: TCP
---
apiVersion: v1
kind: Service
metadata:
  name: inference-backend
  namespace: <your-namespace>
spec:
  ports:
    - port: 8000
      targetPort: 8000
      protocol: TCP
```

## Health Probes (for downstream Deployments)

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 30
readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 10
```

## Security Considerations

Port 8000 has **no auth** by default. Before exposing:

1. **Firewall** — allowlist only your K8s cluster egress IPs on the Hetzner firewall or `ufw` on the host
2. **API key** — add `--api-key <secret>` to vLLM flags in `docker-compose.yml`, then pass `api_key="<secret>"` in the OpenAI client
3. **Teleport Machine ID** — run a `tbot` agent in K8s to tunnel through Teleport (ask DevOps). Most secure option, keeps port 8000 unexposed

## Quick Test (from any machine with access)

```bash
curl http://78.46.219.175:8000/health

curl http://78.46.219.175:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Sehyo/Qwen3.5-122B-A10B-NVFP4",
    "messages": [{"role": "user", "content": "What is Ethereum?"}],
    "max_tokens": 100
  }'
```
