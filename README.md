# 🛡️ SheriffSpeech: LLM-Driven Digital Threat Detection in Voice Communication

> **Graduation Project · KFUPM · Term 242**
> Reem Fuad Shareef · Supervised by Dr. Waleed Algobi

---

## 📌 Project Overview

This project presents an **AI-powered pipeline** that automatically detects unsafe or threatening content in spoken audio. The system transcribes speech, analyzes vocal emotion, and applies a large language model (LLM) guard to classify audio as **Safe** or **Unsafe** in real time.

Two systems are evaluated and compared:

| System | Components | Accuracy | Macro F1 |
|--------|-----------|----------|----------|
| **System A** (Baseline) | Whisper ASR → Qwen3Guard | 96.0% | 96.0% |
| **System B** (Full Pipeline) | Whisper ASR → SpeechBrain SER → Fusion Layer → Qwen3Guard | **99.0%** | **99.0%** |

> ✅ **System B achieves 99% accuracy with zero false alarms**, at the cost of +65.4% latency overhead from SpeechBrain emotion fusion.

---

## 🧠 Pipeline Architecture

```
Audio Input (.wav)
       │
       ▼
┌─────────────────┐
│  Whisper  ASR   │  ← Transcribes speech to text
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ SpeechBrain SER │  ← Classifies emotion: Angry / Neutral / Cheerful / Sad
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Fusion Layer   │  ← Combines transcript + emotion context
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Qwen3Guard LLM │  ← Classifies content as Safe / Unsafe
└────────┬────────┘
         │
         ▼
  🟢 SAFE  /  🔴 UNSAFE
```

---

## 📂 Repository Structure

```
SEC619-Threat-Detection/
│
├── 100_Dataset/
│   ├── Audio/                          ← 100 WAV files (50 Unsafe · 50 Safe)
│   ├── Output/
│   │   ├── comparison_summary_report.txt
│   │   ├── comparison_sysA_vs_sysB.csv
│   │   ├── disagreement_cases.csv
│   │   ├── performance_metrics_comparison.csv
│   │   └── results_json/               ← Per-file JSON prediction results
│   ├── build_dataset_100.py            ← Dataset builder (WAV placeholders + Excel)
│   ├── tts_edge_emotional_large.py     ← TTS audio generator (Edge TTS)
│   ├── dataset_100_samples.json        ← Dataset manifest (100 samples)
│   ├── dataset_100_samples.xlsx        ← 6-sheet Excel workbook
│   └── GUIDE.md                        ← Integration & evaluation guide
│
├── requirements.txt                    ← Python dependencies
├── .gitignore                          ← Files to exclude from Git
└── README.md                           ← This file
```

---

## 🗂️ Dataset

The evaluation dataset contains **100 spoken audio samples** with balanced classes:

### Unsafe Categories (50 samples)
| Code | Category |
|------|----------|
| C01_VIO | Violence / Threats |
| C02_ILL | Illegal Activity |
| C03_SEX | Sexual Content |
| C04_PII | Personal Identifiable Information (PII) Exposure |
| C05_SH | Suicide / Self-Harm |
| C06_ETH | Unethical Behavior |
| C07_POL | Political Extremism |
| C08_CPY | Copyright Infringement |
| C09_JBK | Jailbreak Attempts |

### Safe Categories (50 samples)
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
Each sample is assigned one of 4 emotion classes:
- **Angry** (16 samples) — fast rate (+40%), high pitch (+20Hz)
- **Neutral** (55 samples) — slightly slower (-10%), flat tone
- **Cheerful** (20 samples) — moderate rate (+20%), bright voice
- **Sad** (9 samples) — slow rate (-30%), low pitch (-20Hz)

> Audio is synthesized using **Microsoft Edge TTS** (`edge-tts`) with prosody parameters tuned for SpeechBrain emotion detection.

---

## ⚙️ Installation

### Prerequisites
- Python 3.9 or higher
- CUDA-compatible GPU (recommended for Whisper and SpeechBrain)
- Microphone or WAV audio files

### Step 1 — Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/SEC619-Threat-Detection.git
cd SEC619-Threat-Detection
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Download Whisper model
```python
import whisper
whisper.load_model("base")   # or "small", "medium", "large"
```

---

## 🚀 Quick Start

### Build the dataset (create WAV placeholders + Excel manifest)
```bash
cd 100_Dataset
python build_dataset_100.py
```

### Generate real audio via Edge TTS
```bash
python tts_edge_emotional_large.py
```
> This replaces silent WAV placeholders with emotionally-varied synthesized speech. Generates 100 files (~5–10 minutes).

### Run the full fusion pipeline (System B)
```bash
python Dataset_WithFusionLayer_ColorfulOutput.py
```

### Run evaluation comparison (System A vs System B)
```bash
python evaluate_compare.py
```

### Latency analysis
```bash
python latency_analysis.py
```

---

## 📊 Results

### Performance Metrics

| Metric | System A (Baseline) | System B (Fusion) | Delta |
|--------|--------------------|--------------------|-------|
| Accuracy | 96.0% | **99.0%** | +3.0% ▲ |
| Precision (Unsafe) | 92.6% | **100.0%** | +7.4% ▲ |
| Recall (Unsafe) | 100.0% | 98.0% | -2.0% |
| F1 (Unsafe) | 96.2% | **99.0%** | +2.8% ▲ |
| Precision (Safe) | 100.0% | 98.0% | -2.0% |
| Recall (Safe) | 92.0% | **100.0%** | +8.0% ▲ |
| Macro-avg F1 | 96.0% | **99.0%** | +3.0% ▲ |

### Latency Breakdown

| Stage | System A | System B | Overhead |
|-------|----------|----------|----------|
| Whisper ASR | 22.54s | 15.92s | -6.62s |
| SpeechBrain SER | — (excluded) | 22.07s | +22.07s |
| Qwen3Guard LLM | 0.94s | 0.84s | -0.10s |
| **End-to-End** | **23.48s** | **38.83s** | **+15.35s (+65.4%)** |

### Confusion Matrices

**System A** (Text-only Baseline)
```
                  Pred: UNSAFE   Pred: SAFE
Actual: UNSAFE       TP = 50       FN = 0
Actual: SAFE          FP = 4      TN = 46
```

**System B** (Full Fusion Pipeline)
```
                  Pred: UNSAFE   Pred: SAFE
Actual: UNSAFE       TP = 49       FN = 1
Actual: SAFE          FP = 0      TN = 50
```

---

## 🔑 Key Findings

- ✅ **Fusion improves accuracy by +3.0%** and eliminates all false alarms (FP → 0)
- ⚠️ **Latency overhead of +65.4%** from SpeechBrain — justified by safety gains
- 🔍 Hardest cases: **empathetic/ambiguous language** (e.g., self-harm framed as support)
- 📌 System A missed **0 threats** but had **4 false alarms**
- 📌 System B had **0 false alarms** but missed **1 copyright-infringement case**

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| Speech-to-Text (ASR) | [OpenAI Whisper](https://github.com/openai/whisper) |
| Emotion Recognition (SER) | [SpeechBrain](https://speechbrain.github.io/) |
| LLM Safety Guard | [Qwen3Guard](https://huggingface.co/Qwen) |
| TTS Audio Synthesis | [Microsoft Edge TTS](https://github.com/rany2/edge-tts) |
| Dataset Format | JSON · XLSX · WAV |
| Analysis Output | CSV · TXT Reports |

---

## 👩‍💻 Author

**Reem Fuad Shareef**
King Fahd University of Petroleum and Minerals (KFUPM)

**Supervisor:** Dr. Waleed Algobi
**Course:** SEC619 — Graduation Project

---

## 📄 License

This project is submitted as an academic graduation project at KFUPM. All rights reserved © 2026.
