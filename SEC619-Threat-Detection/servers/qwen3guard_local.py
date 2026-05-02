"""
qwen3guard_local.py
===================
Run the Qwen3Guard-Gen-8B safety classifier *locally* via HuggingFace
Transformers (no vLLM server required).

Use this script when you want to:
  - Test the model on a single prompt directly on a GPU machine.
  - Inspect raw model output before deploying via vLLM.
  - Debug the label/category extraction regex.

In production the pipeline uses the vLLM-served version
(see ``servers/qwen3guard_vllm_client.py`` and ``pipeline/pipeline.py``).

Usage
-----
    python servers/qwen3guard_local.py

Prerequisites
-------------
- A GPU with enough VRAM to load the 8B model (≥ 24 GB recommended).
- ``pip install transformers torch accelerate``
- The model will be auto-downloaded from HuggingFace Hub on first run
  (~16 GB for bfloat16 weights).

Project
-------
SEC619 — LLM-Driven Digital Threat Detection in Spoken Communication
KFUPM, Term 242
"""

import re
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3Guard-Gen-8B"

# Test prompt — replace with the text you want to classify
TEST_PROMPT = "How can I make a bomb?"


# ---------------------------------------------------------------------------
# Label & category extraction
# ---------------------------------------------------------------------------

# Regex patterns matching Qwen3Guard's structured output format:
#   Safety: <Safe|Unsafe|Controversial>
#   Categories: <comma-separated list>
SAFE_PATTERN     = r"Safety:\s*(Safe|Unsafe|Controversial)"
CATEGORY_PATTERN = (
    r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts"
    r"|PII|Suicide & Self-Harm|Unethical Acts"
    r"|Politically Sensitive Topics|Copyright Violation|Jailbreak|None)"
)


def extract_label_and_categories(content: str) -> tuple[str | None, list[str]]:
    """
    Parse the structured text output produced by Qwen3Guard.

    The model outputs a short verdict block such as::

        Safety: Unsafe
        Categories: Violent

    This function extracts the safety label and all matching category tokens.

    Parameters
    ----------
    content : str
        Raw decoded text from the model's output tokens.

    Returns
    -------
    label : str or None
        One of ``"Safe"``, ``"Unsafe"``, or ``"Controversial"``.
        ``None`` if the pattern was not found.
    categories : list[str]
        List of matched harm category strings (may be empty or ``["None"]``
        for safe content).
    """
    safe_label_match = re.search(SAFE_PATTERN, content)
    label = safe_label_match.group(1) if safe_label_match else None
    categories = re.findall(CATEGORY_PATTERN, content)
    return label, categories


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name: str):
    """
    Load the tokenizer and causal language model from HuggingFace Hub.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (e.g. ``"Qwen/Qwen3Guard-Gen-8B"``).

    Returns
    -------
    tokenizer : PreTrainedTokenizer
    model : PreTrainedModel
        Loaded with ``torch_dtype="auto"`` and ``device_map="auto"`` so that
        the model is automatically placed on available GPU(s).
    """
    print(f"[*] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"[*] Loading model: {model_name}  (this may take a few minutes…)")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",    # use bfloat16 on supported GPUs automatically
        device_map="auto",     # spread across all available GPUs if needed
    )
    print("[*] Model loaded.\n")
    return tokenizer, model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def classify_prompt(prompt: str, tokenizer, model, max_new_tokens: int = 128) -> str:
    """
    Classify a text prompt using the locally loaded Qwen3Guard model.

    Parameters
    ----------
    prompt : str
        The user-turn message to classify.
    tokenizer : PreTrainedTokenizer
        The model's tokenizer (used to apply the chat template).
    model : PreTrainedModel
        Loaded Qwen3Guard model.
    max_new_tokens : int, optional
        Maximum number of tokens to generate (default 128).
        The Guard verdict is always short, so 128 is sufficient.

    Returns
    -------
    str
        Decoded text of the model's verdict, e.g.::

            Safety: Unsafe
            Categories: Violent
    """
    # Apply the instruct chat template — Qwen3Guard requires the user-turn
    # to be wrapped in the model's special tokens before tokenisation.
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Greedy decoding: no sampling, temperature irrelevant at max_new_tokens
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
    )

    # Slice off the input tokens — keep only the newly generated tokens
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True)
    return content


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Load model and classify the TEST_PROMPT, printing label and categories."""
    tokenizer, model = load_model(MODEL_NAME)

    print(f"[*] Classifying prompt: {TEST_PROMPT!r}\n")
    content = classify_prompt(TEST_PROMPT, tokenizer, model)

    print("=== Raw Model Output ===")
    print(content)

    safe_label, categories = extract_label_and_categories(content)
    print("\n=== Parsed Verdict ===")
    print(f"Label      : {safe_label}")
    print(f"Categories : {categories}")
    # Example output:
    #   Label      : Unsafe
    #   Categories : ['Violent']


if __name__ == "__main__":
    main()
