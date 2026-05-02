"""
qwen3guard_vllm_client.py
=========================
Quick-test client for the Qwen3Guard-Gen-8B safety classifier served via vLLM.

This script sends a single prompt to the vLLM-hosted Qwen3Guard model using
the OpenAI-compatible REST API and prints the raw JSON response together with
the extracted Guard classification verdict.

Usage
-----
    python servers/qwen3guard_vllm_client.py

Prerequisites
-------------
- Qwen3Guard-Gen-8B must be running on SERVER_B via vLLM (port 8000).
  See servers/server_startup.sh for the launch command.
- `pip install requests`

Configuration
-------------
Replace ``YOUR_SERVER_B_IP`` below with the actual IP address of the GPU
server hosting the Qwen3Guard model before running.

Project
-------
SEC619 — LLM-Driven Digital Threat Detection in Spoken Communication
KFUPM, Term 242
"""

import json
import requests

# ---------------------------------------------------------------------------
# Configuration — update SERVER_B_IP to match your vLLM GPU server
# ---------------------------------------------------------------------------
SERVER_B_IP = "YOUR_SERVER_B_IP"       # e.g. "38.x.x.x"
QWEN_URL    = f"http://{SERVER_B_IP}:8000/v1/chat/completions"
MODEL_NAME  = "Qwen/Qwen3Guard-Gen-8B"

# ---------------------------------------------------------------------------
# Request headers — vLLM accepts any non-empty Bearer token
# ---------------------------------------------------------------------------
HEADERS = {
    "Authorization": "Bearer EMPTY",
    "Content-Type":  "application/json",
}

# ---------------------------------------------------------------------------
# Test prompt — replace with any text you want to classify
# ---------------------------------------------------------------------------
TEST_PROMPT = "I want to make malware"

def classify_prompt(prompt: str) -> dict:
    """
    Send a single text prompt to the vLLM-served Qwen3Guard model and return
    the parsed JSON response.

    Parameters
    ----------
    prompt : str
        The user message to be safety-classified.

    Returns
    -------
    dict
        Full JSON response from the vLLM server, including the Guard verdict
        inside ``choices[0].message.content``.

    Notes
    -----
    - ``temperature=0`` ensures deterministic (greedy) classification.
    - ``verify=False`` disables SSL certificate verification — suitable for
      internal GPU servers with self-signed or no TLS certificates.
    """
    payload = {
        "model":       MODEL_NAME,
        "messages":    [{"role": "user", "content": prompt}],
        "temperature": 0,              # greedy decoding for reproducible verdicts
    }

    response = requests.post(
        QWEN_URL,
        headers=HEADERS,
        json=payload,
        verify=False,                  # skip SSL check for internal server
        timeout=60,
    )
    response.raise_for_status()        # raise HTTPError on 4xx / 5xx
    return response.json()


def main():
    """Entry point: classify TEST_PROMPT and print the Guard verdict."""
    print(f"[*] Sending prompt to Qwen3Guard @ {QWEN_URL}")
    print(f"[*] Prompt: {TEST_PROMPT!r}\n")

    data = classify_prompt(TEST_PROMPT)

    # Pretty-print full JSON response (useful for debugging)
    print("=== Full API Response ===")
    print(json.dumps(data, indent=2))

    # Extract and display only the Guard classification verdict
    guard_output = data["choices"][0]["message"]["content"]
    print("\n=== Guard Classification ===")
    print(guard_output)
    # Expected output format:
    #   Safety: Unsafe
    #   Categories: Violent


if __name__ == "__main__":
    main()
