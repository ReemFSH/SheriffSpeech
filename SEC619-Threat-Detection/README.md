# 🛡️ SheriffSpeech: LLM-Driven Digital Threat Detection in Spoken Communication

> **Graduation Project · King Fahd University of Petroleum and Minerals (KFUPM)**
>
> **Author:** Reem Fuad Shareef 
> **Supervisor:** Dr. Waleed Algobi

---

## 📌 Overview

This project implements a **multimodal AI pipeline** that automatically detects unsafe or threatening content in spoken audio. The system transcribes speech, classifies vocal emotion, fuses both signals, and applies a large language model safety guard to classify each audio clip as **Safe** or **Unsafe**.

Two systems are built and compared to measure the impact of emotion fusion:

| | System A (Baseline) | System B (Full Pipeline) |
|---|---|---|
| **Components** | Whisper → Qwen3Guard | Whisper → SpeechBrain → Fusion Layer → Qwen3Guard |
| **Accuracy** | 96.0% | **99.0%** |
| **Macro F1** | 96.0% | **99.0%** |
| **False Alarms (FP)** | 4 | **0** |
| **End-to-End Latency** | 23.48s | 38.83s (+65.4%) |

> ✅ **System B achieves 99% accuracy with zero false alarms** by adding vocal emotion context to the LLM guard input. The +65.4% latency overhead from SpeechBrain is justified by the +3% macro F1 gain and elimination of all false alarms.

---

## 🧠 Pipeline Architecture

```
Audio Input (.wav)
       │
       ▼
┌──────────────────────┐
│   Whisper Large-v3   │  Stage 1 — ASR
│   Speech-to-Text     │  Transcribes WAV → plain text
│   (Server A, :8001)  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  SpeechBrain SER     │  Stage 2 — Emotion Recognition
│  wav2vec2-IEMOCAP    │  Classifies: Angry / Neutral / Cheerful / Sad
│  (Server A, :9100)   │
└──────────┬───────────┘
           │ (emotion label + confidence scores)
           ▼
┌──────────────────────┐
│   Fusion Layer       │  Bridge — Natural Language Encoding
│   Emotion Prefix     │  "[Audio context: The speaker sounds very angry (Angry=0.87...)]"
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Qwen3Guard-Gen-8B   │  Stage 3 — LLM Safety Classification
│  (Server B, :8000)   │  Classifies: Safe / Controversial / Unsafe
└──────────┬───────────┘  Identifies: threat category from 9-class taxonomy
           │
           ▼
    🟢 SAFE  /  🔴 UNSAFE
```

---

## 📂 Repository Structure

```
SEC619-Threat-Detection/
│
├── pipeline/
│   └── pipeline.py                  ← Main evaluation pipeline (Stage 1-2-3)
│
├── evaluation/
│   ├── compare_systems.py           ← System A vs B metrics comparison
│   ├── visualize_results.py         ← 9 publication-ready comparison charts
│   ├── latency_analysis.py          ← Per-stage latency breakdown table
│   └── notebooks/
│       ├── Evaluation_Baseline_Fusion.ipynb
│       └── SystemA_vs_SystemB_Fusion.ipynb
│
├── servers/
│   ├── speechbrain_api.py           ← FastAPI server for SpeechBrain SER
│   ├── server_startup.sh            ← GPU server startup commands
│   ├── qwen3guard_vllm_client.py    ← Quick-test client for Qwen3Guard via vLLM
│   ├── qwen3guard_local.py          ← Run Qwen3Guard locally via HuggingFace
│   ├── whisper_vllm_client.py       ← Quick-test client for Whisper via vLLM
│   └── whisper_local.py             ← Batch-transcribe locally via HuggingFace
│
├── demo/
│   ├── app.py                       ← Flask demo server (SheriffSpeech)
│   ├── static/
│   │   └── index.html               ← Browser UI (drag-and-drop demo)
│   ├── architecture.html            ← Interactive pipeline architecture diagram
│   └── README.md                    ← Demo run instructions & API reference
│
├── dataset/
│   ├── audio/                       ← 100 WAV audio files (50 Unsafe · 50 Safe)
│   ├── dataset_100_samples.json     ← Dataset manifest (ground truth labels)
│   ├── dataset_100_samples.xlsx     ← 6-sheet Excel workbook
│   ├── build_dataset.py             ← Dataset builder (WAV placeholders + Excel)
│   ├── tts_generator.py             ← TTS audio generator (Edge TTS)
│   └── GUIDE.md                     ← Integration and evaluation guide
│
├── results/
│   ├── output_csv/
│   │   ├── multimodal_result.csv             ← System A raw results
│   │   ├── multimodal_result_fusion.csv      ← System B raw results
│   │   ├── comparison_sysA_vs_sysB.csv       ← Merged per-file comparison
│   │   ├── disagreement_cases.csv            ← Files where systems disagree
│   │   ├── performance_metrics_comparison.csv ← All metrics side-by-side
│   │   └── comparison_summary_report.txt     ← Full text report
│   └── charts/
│       ├── 01_safety_distribution.png
│       ├── 02_agreement_pie.png
│       ├── 03_disagreement_breakdown.png
│       ├── 04_latency_comparison.png
│       ├── 05_category_distribution.png
│       ├── 06_emotion_impact.png
│       ├── 07_performance_metrics.png
│       ├── 08_confusion_matrices.png
│       └── 09_improvement_breakdown.png
│
├── requirements.txt                 ← Python dependencies
├── .gitignore                       ← Git exclusion rules
└── README.md                        ← This file
```

---

## 🗂️ Dataset

The evaluation dataset contains **100 spoken audio samples** with balanced classes, synthesized using Microsoft Edge TTS with emotion-varied prosody parameters.

### Unsafe Categories — 50 samples

| Code | Category | Description |
|------|----------|-------------|
| C01_VIO | Violence / Threats | Explicit threats, intimidation |
| C02_ILL | Illegal Activity | Solicitation, criminal instructions |
| C03_SEX | Sexual Content | Explicit or inappropriate sexual speech |
| C04_PII | PII Exposure | Sharing or soliciting personal data |
| C05_SH | Suicide / Self-Harm | Harmful content framed as advice or support |
| C06_ETH | Unethical Behavior | Deception, fraud, manipulation |
| C07_POL | Political Extremism | Extremist ideology, incitement |
| C08_CPY | Copyright Infringement | Piracy, unauthorized sharing |
| C09_JBK | Jailbreak Attempts | Prompts designed to bypass safety systems |

### Safe Categories — 50 samples

| Code | Category |
|------|----------|
| S01_NPC | Normal Personal Conversation |
| S02_PRO | Professional Discussion |
| S03_SOC | Social Interaction |
| S04_CUS | Customer Service |
| S05_EMO | Emotional Support |
| S06_ETH | Workplace Ethics |
| S07_CIV | Civic Discussion |
| S08_COM | Commerce |
| S09_TEC | Technical Assistance |
| S10_ENT | Entertainment |

### Emotion Distribution

| Emotion | Samples | TTS Parameters |
|---------|---------|----------------|
| Neutral | 55 | Rate: -10%, Pitch: -5Hz |
| Cheerful | 20 | Rate: +20%, Pitch: +10Hz, JennyNeural voice |
| Angry | 16 | Rate: +40%, Pitch: +20Hz |
| Sad | 9 | Rate: -30%, Pitch: -20Hz |

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.9+
- Two GPU servers (for running Whisper, SpeechBrain, and Qwen3Guard)
- Alternatively: run all models locally on a single GPU machine with sufficient VRAM

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/SEC619-Threat-Detection.git
cd SEC619-Threat-Detection
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Start the model servers

See `servers/server_startup.sh` for the exact commands. Summary:

**Server A** (runs Whisper + SpeechBrain):
```bash
# Terminal 1 — Whisper ASR
source ~/whisper_vllm/bin/activate
vllm serve openai/whisper-large-v3 --host 0.0.0.0 --port 8001

# Terminal 2 — SpeechBrain Emotion API
source ~/venvs/voiceguard/bin/activate
uvicorn servers.speechbrain_api:app --host 0.0.0.0 --port 9100
```

**Server B** (runs Qwen3Guard):
```bash
source bin/activate
vllm serve Qwen/Qwen3Guard-Gen-8B --host 0.0.0.0 --port 8000 --max-model-len 8192
```

### Step 4 — Configure server addresses

Edit `pipeline/pipeline.py` → Section D:
```python
WHISPER_SERVER_IP = "YOUR_SERVER_A_IP"
QWENGUARD_URL     = "http://YOUR_SERVER_B_IP:8000/v1/chat/completions"
```

---

## 🚀 Running the Pipeline

### Generate audio dataset (if not pre-generated)

```bash
# Build WAV placeholders + Excel manifest
cd dataset
python build_dataset.py

# Replace placeholders with real TTS audio
python tts_generator.py
```

### Run System A — Baseline (text-only)

```python
# In pipeline/pipeline.py, set:
FUSION_ENABLED = False
CSV_NAME       = "multimodal_result.csv"
```
```bash
python pipeline/pipeline.py
```

### Run System B — Full Fusion Pipeline

```python
# In pipeline/pipeline.py, set:
FUSION_ENABLED = True
CSV_NAME       = "multimodal_result_fusion.csv"
```
```bash
python pipeline/pipeline.py
```

### Compare results

```bash
python evaluation/compare_systems.py
```

### Generate charts

```bash
python evaluation/visualize_results.py
```

### Latency analysis

```bash
python evaluation/latency_analysis.py
```

---

## 🖥️ Interactive Demo

A browser-based demo app (**SheriffSpeech**) lets you drop any audio file and
see the full pipeline run live — transcript, emotion tone, fused prompt, and
Qwen3Guard safety verdict — in under 5 seconds.

```bash
# Install demo dependencies
pip install flask requests openai

# Start the demo server (configure server IPs in demo/app.py first)
python demo/app.py

# Open in browser
http://localhost:5000
```

The demo includes:
- **Drop-zone UI** — drag & drop any WAV/MP3/M4A file
- **Mode toggle** — switch between System A (text-only) and System B (fusion)
- **Live latency panel** — per-stage breakdown (ASR / SER / Guard)
- **Server health check** — verify all GPU services before running
- **Architecture diagram** — open `demo/architecture.html` in any browser

See [`demo/README.md`](demo/README.md) for full API documentation and run instructions.

---

## 📊 Results

### Performance Comparison

| Metric | System A (Baseline) | System B (Fusion) | Δ |
|--------|--------------------|--------------------|---|
| Accuracy | 96.0% | **99.0%** | +3.0% ▲ |
| Precision (Unsafe) | 92.6% | **100.0%** | +7.4% ▲ |
| Recall (Unsafe) | **100.0%** | 98.0% | -2.0% |
| F1 (Unsafe) | 96.2% | **99.0%** | +2.8% ▲ |
| Precision (Safe) | **100.0%** | 98.0% | -2.0% |
| Recall (Safe) | 92.0% | **100.0%** | +8.0% ▲ |
| **Macro-avg F1** | 96.0% | **99.0%** | **+3.0% ▲** |

### Confusion Matrices

**System A — Baseline**
```
                   Pred: UNSAFE   Pred: SAFE
Actual: UNSAFE        TP = 50       FN = 0   ← caught all threats
Actual: SAFE           FP = 4      TN = 46   ← 4 false alarms
```

**System B — Full Fusion Pipeline**
```
                   Pred: UNSAFE   Pred: SAFE
Actual: UNSAFE        TP = 49       FN = 1   ← 1 missed threat
Actual: SAFE           FP = 0      TN = 50   ← zero false alarms ✅
```

### Corrected Latency Breakdown

| Stage | System A | System B | Overhead |
|-------|----------|----------|----------|
| Whisper ASR | 22.54s | 15.92s | -6.62s |
| SpeechBrain SER | — (excluded) | 22.07s | +22.07s |
| Qwen3Guard LLM | 0.94s | 0.84s | -0.10s |
| **End-to-End** | **23.48s** | **38.83s** | **+15.35s (+65.4%)** |

> Note: System A latency excludes SpeechBrain time because its output was discarded in the baseline. See `evaluation/compare_systems.py` for the full latency correction methodology.

---

## 🔑 Key Findings

- ✅ **Emotion fusion improves Macro F1 by +3% and eliminates all false alarms**
- ⚠️ **65.4% latency overhead** from SpeechBrain — justified by safety-critical gains
- 🔍 **Hardest case**: self-harm content framed as empathetic/supportive speech
- 📊 System B disagreed with System A on **5 files**: it correctly resolved **4 of 5**

---

## 🛠️ Technology Stack

| Component | Model / Tool | Version |
|-----------|-------------|---------|
| ASR (Speech-to-Text) | [OpenAI Whisper Large-v3](https://github.com/openai/whisper) | large-v3 |
| SER (Emotion Recognition) | [SpeechBrain wav2vec2-IEMOCAP](https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP) | — |
| LLM Safety Guard | [Qwen3Guard-Gen-8B](https://huggingface.co/Qwen/Qwen3Guard-Gen-8B) | 8B params |
| Model Serving | [vLLM](https://github.com/vllm-project/vllm) | — |
| TTS Synthesis | [Microsoft Edge TTS](https://github.com/rany2/edge-tts) | — |
| API Framework | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn | — |
| Data Processing | pandas, numpy | — |
| Visualization | matplotlib, seaborn | — |

---

## 📈 Results Charts

All 9 comparison charts are saved in `results/charts/`:

| Chart | Description |
|-------|-------------|
| `01_safety_distribution.png` | Safe vs Unsafe prediction counts |
| `02_agreement_pie.png` | System agreement rate (95%) |
| `03_disagreement_breakdown.png` | Classification outcome buckets |
| `04_latency_comparison.png` | Per-stage latency comparison |
| `05_category_distribution.png` | Top threat categories detected |
| `06_emotion_impact.png` | How emotion class affects classification |
| `07_performance_metrics.png` | All metrics side-by-side with deltas |
| `08_confusion_matrices.png` | Side-by-side heatmaps |
| `09_improvement_breakdown.png` | Predictions fixed vs broken by fusion |

---

## 👩‍💻 Author & Acknowledgements

**Reem Fuad Shareef**
King Fahd University of Petroleum and Minerals (KFUPM)

**Supervisor:** Dr. Waleed Algobi
**Course:** SEC619 — Graduation Project

---

## 📄 License

This project is submitted as an academic graduation project at KFUPM.
All rights reserved © 2026 Reem Fuad Shareef.
