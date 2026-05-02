# SEC619 — Integration and Evaluation Guide
## LLM-Driven Digital Threat Detection in Spoken Communication
**Reem Fuad Shareef · Dr. Waleed Algobi · Term 242**

---

## What Is in This Package

This package contains five files that work together as a complete evaluation system for your graduation project:

| File | Purpose |
|---|---|
| `Dataset_WithFusionLayer_ColorfulOutput.py` | Main pipeline — 967 lines, all integrations applied |
| `tts_generator_combined.py` | Generates real speech audio from dataset scripts |
| `evaluate_compare.py` | Two-run comparison: System A vs System B |
| `latency_analysis.py` | Latency breakdown table for Section 7.5 |
| `GUIDE.md` | This file — step-by-step instructions |

The dataset files (`dataset_combined_60.json`, `audio/`) come from your existing `dataset_v2` folder.

---

## Section 1 — What Was Changed in the Pipeline

The pipeline file has been extended with six integration blocks on top of the fusion layer that was already implemented. Nothing in the existing transcription, emotion, or guard logic was changed. The additions are:

**ANSI colour class** (line 35) — enables green/red console output when predictions match or miss ground truth. No external library required.

**Ground truth loader** (line 51) — reads `dataset_combined_60.json` at startup and builds a dictionary mapping each WAV filename to its correct label. The lookup key is the filename only (e.g. `C01_VIO_S01.wav`), so it works regardless of which folder the audio files are in.

**`FUSION_ENABLED` flag** (line 126) — a single boolean that controls whether SpeechBrain emotion features are prepended to the Qwen3Guard input. Set it to `False` for the baseline run and `True` for the full pipeline run. This is the only thing you change between the two runs.

**Updated `build_fused_input()`** (line 436) — now respects `FUSION_ENABLED`. When `False`, returns the plain cleaned transcript. When `True`, prepends the natural-language emotion context exactly as before.

**Ground truth comparison block** (line 789) — runs after each file completes. Compares the predicted safety label against the manifest ground truth and prints a green tick or red cross. Also increments the TP/TN/FP/FN counters.

**Two new CSV columns** — `ground_truth` and `correct` are now written to every row and included in the JSON bundle. This makes the output CSV self-contained for analysis without needing to re-join the manifest.

**Performance banner** (line 912) — replaces the old simple summary. After all 60 files are processed, prints accuracy, precision, recall, F1 for both classes, macro F1, and a warning if any threats were missed.

---

## Section 2 — Setup Steps

### Step 1 — Extract the dataset

Unzip `dataset_combined_60.zip` to a folder on your machine. The structure should be:

```
C:\dataset_v2\
├── audio\
│   ├── C01_VIO_S01.wav
│   ├── C01_VIO_S02.wav
│   └── ... (60 WAV files total)
├── dataset_combined_60.json
├── tts_generator_combined.py
└── ...
```

The 60 WAV files in `audio\` are currently silent placeholders (2-second silence). Before running the pipeline, you must replace them with real TTS speech.

### Step 2 — Generate real audio

Install the two TTS dependencies:

```
pip install gTTS pydub
```

Then run the TTS generator from inside the `dataset_v2` folder:

```
cd C:\dataset_v2
python tts_generator_combined.py --engine gtts --lang en --delay 2.0
```

This reads each script from `dataset_combined_60.json` and writes a real 16 kHz mono WAV file to `audio\`. The `--delay 2.0` flag adds a 2-second pause between files to avoid hitting Google TTS rate limits. Generating all 60 files takes approximately 5 to 10 minutes depending on your connection.

If Google TTS is unavailable, use the local engine instead:

```
pip install pyttsx3
python tts_generator_combined.py --engine pyttsx3
```

The local engine is faster but produces a more robotic voice. Both engines produce valid 16 kHz WAV files that Whisper accepts directly.

### Step 3 — Update paths in the pipeline file

Open `Dataset_WithFusionLayer_ColorfulOutput.py` and update four lines near the top:

```python
# Line 71 — folder containing the 60 WAV files
INPUT_DIR = Path(r"C:\dataset_v2\audio")

# Line 74 — folder where results will be written
OUT_DIR = Path(r"C:\dataset_v2\output_results")

# Line 51 — path to the JSON manifest
MANIFEST_PATH = Path(r"C:\dataset_v2\dataset_combined_60.json")
```

The `CSV_NAME` line controls the output filename. Leave it as shown — the comments tell you which name to use for each run.

### Step 4 — Confirm servers are running

Your pipeline uses three GPU server endpoints. Confirm all three are up before starting a run:

| Service | Server | Port | Check command |
|---|---|---|---|
| Whisper ASR | Server A | 8001 | `curl http://SERVER_A_IP:8001/v1/models` |
| SpeechBrain Emotion | Server A | 9100 | `curl http://SERVER_A_IP:9100/health` |
| Qwen3Guard | Server B | 8000 | `curl http://SERVER_B_IP:8000/v1/models` |

If any server is not responding, the pipeline will retry 3 times then mark the file as failed and continue. It will not crash.

---

## Section 3 — Running the Two-System Evaluation

Your supervisor requires a direct comparison between two configurations. You run the pipeline twice on the same 60 audio files — only `FUSION_ENABLED` and `CSV_NAME` change between runs.

### Run 1 — System A (baseline, text-only)

In the pipeline file, set:
```python
FUSION_ENABLED = False
CSV_NAME = "multimodal_result.csv"
```

Run the pipeline:
```
python Dataset_WithFusionLayer_ColorfulOutput.py
```

This produces `output_results\multimodal_result.csv`. System A sends only the Whisper transcript to Qwen3Guard — no emotion context.

### Run 2 — System B (full fusion pipeline)

Change only these two lines:
```python
FUSION_ENABLED = True
CSV_NAME = "multimodal_result_fusion.csv"
```

Run the pipeline again:
```
python Dataset_WithFusionLayer_ColorfulOutput.py
```

This produces `output_results\multimodal_result_fusion.csv`. System B prepends the SpeechBrain emotion context to every transcript before sending it to Qwen3Guard.

Note: SpeechBrain runs in both modes. This is intentional — it keeps the latency measurement consistent and ensures the only variable between runs is the fusion input, not which services were called.

### Step 3 — Generate the comparison report

After both runs complete:

```
python evaluate_compare.py
```

This prints the full side-by-side metrics table (Table 4 in your report), both confusion matrices (Table 5), the latency breakdown, and a list of any missed threats.

For the latency table specifically:

```
python latency_analysis.py
```

---

## Section 4 — Understanding the Output

### CSV columns

Your output CSV now has 23 columns. The two new ones at the end are:

`ground_truth` — the correct label from the manifest (`Unsafe` or `Safe`). Empty if the filename was not found in the manifest.

`correct` — `Y` if the pipeline's prediction matches the ground truth, `N` if it does not, empty if ground truth was not available.

### Console output

During each run you will see:
- A green `✓ CORRECT` line for each correct prediction
- A red `✗ WRONG` line for each incorrect prediction, showing both the ground truth and the predicted label
- A full performance table at the end of the run

### JSON bundles

Each per-file JSON now includes `"ground_truth"`, `"correct"`, and `"fusion_enabled"` fields. This makes individual file results fully self-contained for debugging.

---

## Section 5 — Report Writing Guide

Use the values from `evaluate_compare.py` to fill your report sections. The mapping is:

**Section 7.1 — Experimental Setup**
Describe the two-server TensorDock deployment (Server A: 8001 + 9100, Server B: 8000), the dataset (60 samples, 30 unsafe / 30 safe, 19 categories), and the two run configurations. State that `STRICT_MODE=True` meaning Controversial is treated as Unsafe, and `Language=English`, `Temperature=0.0` for deterministic transcription.

**Section 7.2 — Dataset Statistics**
Report the totals: 60 samples, 50/50 split, 9 unsafe categories (all Qwen3Guard categories), 10 safe categories mirroring the same communication channels. State that audio is synthetic TTS (gTTS or pyttsx3), 16 kHz mono WAV, and that ground truth was manually annotated in the JSON manifest.

**Section 7.3 — Results (Full Pipeline)**
Take the System B column from `evaluate_compare.py` output. Present the confusion matrix. In your discussion, address the TP/FN balance: in a cybersecurity system, a missed threat (FN) is more dangerous than a false alarm (FP). Discuss whether your FN count is acceptable.

**Section 7.4 — Baseline Comparison**
Present Table 10 (the full side-by-side with deltas). For each metric where Delta is positive, explain why fusion helped. Focus on Recall and F1 for the Unsafe class — these are your primary claims. If Delta on any metric is negative, acknowledge it honestly and explain the possible cause (for example, emotion features may add noise for categories like Jailbreak where tone is typically neutral regardless of intent).

**Section 7.5 — Latency Analysis**
Use the output of `latency_analysis.py`. ASR (Whisper) will typically dominate — this is expected because audio-to-text conversion is computationally heavier than the guard classification. Quantify the fusion overhead: the difference in `tone_seconds` between a baseline run and a fusion run is the cost of SpeechBrain; the `guard_seconds` may also increase slightly because the fused input is longer.

**Section 7.6 — Discussion**
Address three points: (1) which threat categories benefit most from emotion fusion — violent acts, unethical acts, and suicide/self-harm all have emotionally distinctive delivery patterns; (2) where the system struggles — jailbreak samples often sound calm and neutral, so emotion is uninformative for that category; (3) the key limitation — TTS-generated audio does not capture the full emotional range of real human speech, meaning SpeechBrain emotion scores may be less informative than they would be with real recordings. This is a known limitation to state explicitly and address in future work.

---

## Section 6 — Metric Formulas Reference

These formulas are used by both `evaluate_compare.py` and in your report. Unsafe is the positive class.

```
Accuracy       = (TP + TN) / (TP + TN + FP + FN)

Precision (U)  = TP / (TP + FP)
Recall    (U)  = TP / (TP + FN)
F1        (U)  = 2 × Precision × Recall / (Precision + Recall)

Precision (S)  = TN / (TN + FN)
Recall    (S)  = TN / (TN + FP)
F1        (S)  = 2 × Precision_S × Recall_S / (Precision_S + Recall_S)

Macro F1       = (F1_Unsafe + F1_Safe) / 2
```

The Macro F1 is the single number to report in your abstract and conclusion. It is class-balanced and appropriate for an even 50/50 dataset.

False Negatives (FN) is the most important individual cell in your confusion matrix. In a cybersecurity context, a missed threat is more costly than a false alarm. Your evaluation discussion should explicitly address whether the fusion layer reduced FN compared to the baseline.

---

## Section 7 — Troubleshooting

**Ground truth dict is empty after startup**
Check that `MANIFEST_PATH` points to `dataset_combined_60.json` and that the file exists at that path. The manifest must be the combined 60-sample version, not the earlier `dataset_manifest_v2.json` (which contains only the 30 unsafe samples).

**`correct` column is empty in CSV**
This means the filename was not found in the ground truth dictionary. Check that the WAV filenames in your `audio\` folder exactly match the `id` fields in the JSON (e.g. `C01_VIO_S01.wav`, not `c01_vio_s01.wav`). The lookup is case-sensitive on Linux servers.

**SpeechBrain returns empty tone dict**
This is handled gracefully — `build_fused_input` falls back to plain transcript automatically. The file will still be classified; the status will be `WARN_TONE_FAILED` rather than `OK`. These rows are included in the CSV but excluded from latency statistics.

**Qwen3Guard returns `Controversial` but CSV shows `Unsafe`**
This is correct behaviour. `STRICT_MODE=True` promotes Controversial to Unsafe in `normalize_safety_label()`. The original guard output is preserved in `guard_raw_full` if you need to review it.

**TTS generation fails for some files**
gTTS is a network call to Google's servers. If you hit a rate limit, increase `--delay` (try `--delay 5.0`). Alternatively switch to `--engine pyttsx3` for fully offline generation. Both produce identical 16 kHz mono WAV files.

---

*SEC619 · Information and Computer Science Department · Professional Master of Cybersecurity · April 2026*
