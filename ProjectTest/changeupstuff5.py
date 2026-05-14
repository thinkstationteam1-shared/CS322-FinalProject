import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
from json_repair import repair_json
import random
import argparse
import logging
import time
import re
import signal
from pathlib import Path
from typing import Dict, List, Optional
# transformers 4.46.0 — do NOT import BitsAndBytesConfig; bnb 0.45.0 has no
# GPU support in this environment so we skip quantization entirely and load
# in float16 directly via torch, which CAN see the CUDA device.
from transformers import AutoTokenizer, AutoModelForCausalLM

# Setup logging
logging.basicConfig(format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)
log = logging.getLogger(__name__)

# --- GPU / device probe ---
# torch uses its own CUDA init path, independent of bitsandbytes, so this
# works even when bitsandbytes' NVML init fails.
_cuda_ok = torch.cuda.is_available()
_num_gpus = torch.cuda.device_count() if _cuda_ok else 0
log.info(f"[init] torch.cuda.is_available() = {_cuda_ok}")
log.info(f"[init] GPUs visible to torch: {_num_gpus}")

# Collect per-GPU VRAM so we can set max_memory precisely.
# We leave a 1 GiB headroom per card for CUDA kernels / activations.
_GPU_HEADROOM_GIB = 10
_GPU_CAP_GIB = 10
_gpu_vram: dict = {}
for _i in range(_num_gpus):
    _props = torch.cuda.get_device_properties(_i)
    _total_gib = _props.total_memory / (1024 ** 3)
    _usable_gib = min(_GPU_CAP_GIB, int(_total_gib))
   # _usable_gib = max(1, int(_total_gib) - _GPU_HEADROOM_GIB)
    _gpu_vram[_i] = f"{_usable_gib}GiB"
    log.info(f"[init] GPU {_i}: {_props.name}  total={_total_gib:.1f} GiB  allocated={_usable_gib} GiB")

DEVICE = "cuda" if _cuda_ok else "cpu"
log.info(f"[init] Base device='{DEVICE}' | multi-GPU sharding={'yes' if _num_gpus > 1 else 'no'}")

# --- Vocabulary Levels ---
VOCAB_LEVELS = {
    "A1": 500, "A2": 1000, "B1": 2000, "B2": 4000, "C1": 8000, "C2": 15000
}

# --- Prompts ---
SILVER_SYSTEM = (
    "You are a curriculum designer creating training data for a vocabulary-aware "
    "reading recommender. Given an article passage and vocabulary level, generate "
    "a JSON training example. Output ONLY valid JSON with these keys:\n"
    '  "instruction" : student query describing interests and vocabulary level\n'
    '  "context"     : the article passage\n'
    '  "response"    : object with keys:\n'
    '      "recommended_title"   : string\n'
    '      "summary"             : 2-3 sentence summary\n'
    '      "new_vocabulary"      : list of up to 5 {"word","definition","example"}\n'
    '      "difficulty_rating"   : integer 1-10\n'
    '      "confidence_score"    : float 0.0-1.0\n'
    '      "why_good_next_read"  : 1-2 sentence explanation\n'
    "No markdown, no preamble — JSON only."
)

SILVER_USER = (
    "Article title: {title}\n"
    "Vocabulary level: {level} (~{size} known words)\n"
    "Coverage ratio: {coverage:.1%}\n"
    "New words for this student: {new_words}\n\n"
    "Passage:\n{passage}\n\n"
    "Generate the training example JSON:"
)

# --- Timeout helper (Linux/macOS only) ---
class TimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def repair_json_with_timeout(text: str, timeout_seconds: int = 10) -> str:
    """
    Wraps repair_json with a hard timeout to prevent stalling.
    Falls back to returning the original text if it times out.
    """
    log.debug(f"[repair_json] Starting repair on {len(text)}-char string")
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        result = repair_json(text)
        signal.alarm(0)  # Cancel alarm
        log.debug(f"[repair_json] Completed successfully, result length: {len(result)}")
        return result
    except TimeoutError:
        log.warning(f"[repair_json] TIMED OUT after {timeout_seconds}s — skipping repair, returning original")
        return text
    except Exception as e:
        log.warning(f"[repair_json] Exception during repair: {type(e).__name__}: {e}")
        return text
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class LocalLLM:
    """
    Loads Llama 3.1 in float16 across both GPUs without bitsandbytes.

    Multi-GPU strategy
    ------------------
    accelerate's device_map="auto" with an explicit max_memory dict is the
    correct way to force layer sharding across two cards:

      max_memory = {0: "22GiB", 1: "22GiB", "cpu": "0GiB"}

    The dict is built at runtime from actual VRAM minus a 1 GiB headroom, so
    it adapts to whatever cards are installed.  Setting cpu to "0GiB" tells
    accelerate NOT to spill layers to CPU — if the model doesn't fit purely on
    the two GPUs the load will raise a clear OOM rather than silently running
    10x slower.  Remove that line if you want CPU offload as a last resort.

    Why no 4-bit quant
    ------------------
    bitsandbytes==0.45.0 in this environment was compiled without GPU support
    (NVML init fails), so BitsAndBytesConfig raises a RuntimeError.  fp16
    with device_map sharding is equivalent in throughput and only ~2x the VRAM.
    """

    def __init__(self, model_id: str, token: str, use_half: bool = True):
        log.info(f"[LocalLLM.__init__] Loading model: {model_id}")
        log.info(f"[LocalLLM.__init__] use_half={use_half} | detected GPUs={_num_gpus}")

        dtype = torch.float16 if (use_half and _cuda_ok) else torch.float32
        log.info(f"[LocalLLM.__init__] torch dtype: {dtype}")

        # Build max_memory: use every detected GPU, no CPU spill.
        # _gpu_vram = {0: "22GiB", 1: "22GiB"} computed at module load.
        if _num_gpus >= 2:
            max_memory = {**_gpu_vram, "cpu": "0GiB"}
            log.info(f"[LocalLLM.__init__] Dual-GPU max_memory: {max_memory}")
        elif _num_gpus == 1:
            max_memory = {0: _gpu_vram[0], "cpu": "0GiB"}
            log.info(f"[LocalLLM.__init__] Single-GPU max_memory: {max_memory}")
        else:
            max_memory = None  # CPU-only: let accelerate do whatever
            log.warning("[LocalLLM.__init__] No CUDA GPUs found — falling back to CPU (will be very slow)")

        log.info("[LocalLLM.__init__] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        log.info("[LocalLLM.__init__] Tokenizer loaded. Loading model weights (may take several minutes)...")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            # "auto" + max_memory → accelerate splits layers across GPUs to
            # fill card 0 first, then card 1, respecting the per-device cap.
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True,   # stream weights from disk, not RAM
            trust_remote_code=True,
            token=token,
        )
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            log.debug("[LocalLLM.__init__] pad_token was None, set to eos_token")

        # Log exactly which layers landed on which device so we can verify
        # both GPUs are actually being used.
        hf_device_map = getattr(self.model, "hf_device_map", {})
        if hf_device_map:
            device_counts: dict = {}
            for layer, dev in hf_device_map.items():
                device_counts[str(dev)] = device_counts.get(str(dev), 0) + 1
            log.info(f"[LocalLLM.__init__] Layer distribution across devices: {device_counts}")
            log.info(f"[LocalLLM.__init__] Full device map (first 20 entries): "
                     f"{dict(list(hf_device_map.items())[:20])}")
        else:
            log.info("[LocalLLM.__init__] hf_device_map not available (single-device load)")

        # Report actual VRAM usage post-load
        for _i in range(_num_gpus):
            alloc = torch.cuda.memory_allocated(_i) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(_i) / (1024 ** 3)
            log.info(f"[LocalLLM.__init__] GPU {_i} VRAM after load — "
                     f"allocated={alloc:.2f} GiB  reserved={reserved:.2f} GiB")

    @property
    def _first_device(self) -> torch.device:
        """
        Device that holds the embedding layer (always device 0 with
        device_map='auto').  Input tensors must be on this device.
        """
        return next(self.model.parameters()).device

    def generate(self, system: str, user: str, max_tokens: int = 256) -> Optional[str]:
        log.debug(f"[LocalLLM.generate] Encoding prompt (user msg: {len(user)} chars)")
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]

        encodings = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        # Input must live on the same device as the embedding layer.
        # accelerate moves activations between GPUs internally during the
        # forward pass, so we only need to place the *input* correctly.
        first_dev = self._first_device
        encodings = {k: v.to(first_dev) for k, v in encodings.items()}

        input_len = encodings["input_ids"].shape[-1]
        log.debug(f"[LocalLLM.generate] {input_len} input tokens on {first_dev}. Generating...")

        gen_start = time.time()
        with torch.no_grad():
            output_ids = self.model.generate(
                **encodings,
                max_new_tokens=256,
                min_new_tokens=120,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        gen_elapsed = time.time() - gen_start

        response_ids = output_ids[0][input_len:]
        decoded = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        log.debug(
            f"[LocalLLM.generate] {len(response_ids)} new tokens in {gen_elapsed:.1f}s "
            f"({len(decoded)} chars)"
        )
        if gen_elapsed > 60:
            log.warning(
                f"[LocalLLM.generate] SLOW: {gen_elapsed:.1f}s — "
                "check GPU utilisation with 'nvidia-smi dmon -s u'"
            )

        return decoded


def load_checkpoint(checkpoint_path: Path) -> List[Dict]:
    """Load previously saved examples from a checkpoint file."""
    if not checkpoint_path.exists():
        log.info(f"[checkpoint] No checkpoint found at {checkpoint_path}")
        return []
    examples = []
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    log.warning(f"[checkpoint] Skipping corrupt checkpoint line: {e}")
    log.info(f"[checkpoint] Resumed {len(examples)} examples from {checkpoint_path}")
    return examples


def save_checkpoint(checkpoint_path: Path, examples: List[Dict]):
    """Append a batch of examples to the checkpoint file."""
    with open(checkpoint_path, "a", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    log.debug(f"[checkpoint] Saved {len(examples)} example(s) to {checkpoint_path}")


def generate_example(llm: LocalLLM, article: Dict, level: str) -> Optional[Dict]:
    """Prepare prompt, call LLM, parse and return JSON example."""
    article_id = article.get("id", "unknown")
    log.debug(f"[generate_example] START article_id={article_id} level={level}")

    analysis = article.get("analysis", {})
    new_words = analysis.get("new_words", [])[:10]
    coverage = analysis.get("coverage_ratio", 0.93)

    words = article["text"].split()
    start = random.randint(0, max(0, len(words) - 256))
    passage = " ".join(words[start: start + 256])
    log.debug(f"[generate_example] Passage: words[{start}:{start+256}] of {len(words)} total")

    t0 = time.time()
    raw_response = llm.generate(
        SILVER_SYSTEM,
        SILVER_USER.format(
            title=article["title"],
            level=level,
            size=VOCAB_LEVELS[level],
            coverage=coverage,
            new_words=", ".join(new_words) or "none",
            passage=passage,
        )
    )
    generation_time = time.time() - t0
    log.debug(f"[generate_example] LLM returned in {generation_time:.1f}s")

    if not raw_response:
        log.warning(f"[generate_example] Empty response for article_id={article_id}")
        return None

    # Strip markdown code fences
    clean_json = re.sub(r"```(?:json)?|```", "", raw_response).strip()
    log.debug(f"[generate_example] After strip: {len(clean_json)} chars, starts with: {clean_json[:80]!r}")

    # Attempt repair with timeout guard
    t1 = time.time()
    fixed_json = repair_json_with_timeout(clean_json, timeout_seconds=10)
    repair_time = time.time() - t1
    log.debug(f"[generate_example] repair_json completed in {repair_time:.2f}s")

    try:
        t2 = time.time()
        parsed = json.loads(fixed_json)
        parse_time = time.time() - t2
        log.debug(f"[generate_example] json.loads succeeded in {parse_time:.4f}s")
    except Exception as e:
        log.warning(f"[generate_example] json.loads FAILED for article_id={article_id}: {type(e).__name__}: {e}")
        log.warning(f"[generate_example] fixed_json preview: {fixed_json[:300]!r}")
        log.warning("=" * 60)
        log.warning("RAW MODEL OUTPUT:")
        log.warning(clean_json)
        log.warning("=" * 60)
        return None

    parsed["_meta"] = {
        "article_id": article_id,
        "vocab_level": level,
        "source_title": article["title"]
    }
    log.debug(f"[generate_example] SUCCESS article_id={article_id}")
    return parsed


def generate_batch(
    llm: LocalLLM,
    articles: List[Dict],
    level: str,
    target_count: int,
    checkpoint_path: Path,
    checkpoint_every: int = 50,
    already_have: int = 0,
) -> List[Dict]:
    """
    Generate up to `target_count` examples for `level`, processing `articles`
    in batches. Checkpoints every `checkpoint_every` new successes.
    Returns all newly generated examples (not including pre-loaded ones).
    """
    examples = []
    count = already_have          # total valid so far (including resumed)
    attempts = 0
    stall_streak = 0              # consecutive article batches with 0 successes
    start_time = time.time()

    log.info(f"[generate_batch] Level={level} | target={target_count} | resuming from {already_have}")

    batch_size = 8  # articles processed per inner loop before logging
    article_idx = 0

    while count < target_count and article_idx < len(articles):
        batch = articles[article_idx: article_idx + batch_size]
        article_idx += batch_size
        batch_successes = 0

        log.debug(f"[generate_batch] Processing batch articles[{article_idx-batch_size}:{article_idx}]")

        for art in batch:
            if count >= target_count:
                break

            attempts += 1
            log.debug(f"[generate_batch] Attempt {attempts} — article '{art.get('title', '?')[:60]}'")

            ex = generate_example(llm, art, level)

            if ex:
                examples.append(ex)
                count += 1
                batch_successes += 1
                stall_streak = 0

                # Checkpoint every N new successes
                if len(examples) % checkpoint_every == 0:
                    log.info(f"[generate_batch] Checkpointing {checkpoint_every} examples at count={count}")
                    save_checkpoint(checkpoint_path, examples[-checkpoint_every:])
            else:
                log.debug(f"[generate_batch] Attempt {attempts} failed (None returned)")

        # Batch-level stall detection
        if batch_successes == 0:
            stall_streak += 1
            log.warning(
                f"[generate_batch] Batch yielded 0 successes (stall_streak={stall_streak}). "
                f"attempts={attempts}, count={count}"
            )
            if stall_streak >= 5:
                log.error(
                    f"[generate_batch] 5 consecutive zero-success batches for level={level}. "
                    "This may indicate a stall or data exhaustion. Moving on."
                )
                break
        else:
            stall_streak = 0

        # Periodic progress log
        elapsed = time.time() - start_time
        avg_time = elapsed / attempts if attempts else 0
        success_rate = (count - already_have) / attempts if attempts else 0
        remaining = target_count - count
        eta_min = (remaining / success_rate * avg_time / 60) if success_rate > 0 else float("inf")

        log.info(
            f"[generate_batch] Level={level} | {count}/{target_count} valid | "
            f"{attempts} attempts | {success_rate:.1%} success | "
            f"{avg_time:.1f}s/attempt | ETA {eta_min:.1f} min"
        )

    # Save any remaining unsaved examples
    remainder = len(examples) % checkpoint_every
    if remainder > 0:
        log.info(f"[generate_batch] Final checkpoint: saving last {remainder} examples")
        save_checkpoint(checkpoint_path, examples[-remainder:])

    total_elapsed = time.time() - start_time
    final_rate = (count - already_have) / attempts if attempts else 0
    log.info(
        f"[generate_batch] DONE Level={level}: {count - already_have} new examples "
        f"in {attempts} attempts ({final_rate:.1%} success) "
        f"over {total_elapsed / 60:.1f} min"
    )
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_file", type=str, default="data/corpus/candidates.jsonl")
    parser.add_argument("--output_dir", type=str, default="data/finetune")
    parser.add_argument("--num_pairs", type=int, default=5000)
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--token", type=str, default="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")
    parser.add_argument("--checkpoint_every", type=int, default=50,
                        help="Save a checkpoint every N successful examples per level")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Articles to process per inner batch")
    parser.add_argument("--debug", action="store_true",
                        help="Enable DEBUG-level logging for detailed tracing")
    parser.add_argument("--no_half", action="store_true",
                        help="Load model in float32 instead of float16 (slower, uses more VRAM/RAM)")
    args = parser.parse_args()

    if args.debug:
        log.setLevel(logging.DEBUG)
        log.debug("[main] DEBUG logging enabled")

    log.info(f"[main] Args: {vars(args)}")

    if not os.path.exists(args.corpus_file):
        log.error(f"[main] File {args.corpus_file} not found.")
        return

    candidates = []
    with open(args.corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            candidates.append(json.loads(line))
    log.info(f"[main] Loaded {len(candidates)} candidates.")

    random.seed(42)
    random.shuffle(candidates)

    n_test_articles = max(1, int(len(candidates) * 0.10))
    test_articles = candidates[-n_test_articles:]
    train_articles = candidates[:-n_test_articles]
    log.info(f"[main] Split: {len(train_articles)} train articles, {len(test_articles)} test articles")

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    log.info(f"[main] Output dir: {out_path} | Checkpoint dir: {checkpoint_dir}")

    llm = LocalLLM(args.model_id, args.token, use_half=not args.no_half)
    levels = list(VOCAB_LEVELS.keys())
    per_level = args.num_pairs // len(levels)
    log.info(f"[main] Targeting {per_level} examples per level across {levels}")

    all_train_val = []

    for level in levels:
        checkpoint_path = checkpoint_dir / f"checkpoint_{level}.jsonl"

        # Resume from checkpoint if it exists
        resumed = load_checkpoint(checkpoint_path)
        already_have = len(resumed)
        all_train_val.extend(resumed)

        if already_have >= per_level:
            log.info(f"[main] Level={level} already complete ({already_have} examples). Skipping.")
            continue

        new_examples = generate_batch(
            llm=llm,
            articles=train_articles,
            level=level,
            target_count=per_level,
            checkpoint_path=checkpoint_path,
            checkpoint_every=args.checkpoint_every,
            already_have=already_have,
        )
        all_train_val.extend(new_examples)
        log.info(f"[main] Level={level} complete. all_train_val size: {len(all_train_val)}")

    # Shuffle and train/val split
    log.info(f"[main] Shuffling {len(all_train_val)} examples and splitting train/val...")
    random.shuffle(all_train_val)
    split_idx = int(len(all_train_val) * 0.88)
    train_data = all_train_val[:split_idx]
    val_data = all_train_val[split_idx:]
    log.info(f"[main] train={len(train_data)}, val={len(val_data)}")

    # Generate test examples from held-out articles
    log.info("[main] Generating test examples from held-out articles...")
    test_data = []
    test_checkpoint = checkpoint_dir / "checkpoint_test.jsonl"
    test_resumed = load_checkpoint(test_checkpoint)
    test_data.extend(test_resumed)
    test_target = args.num_pairs // 10

    for art in test_articles:
        if len(test_data) >= test_target:
            break
        level = random.choice(levels)
        log.debug(f"[main] Test article: '{art.get('title','?')[:60]}' level={level}")
        ex = generate_example(llm, art, level)
        if ex:
            test_data.append(ex)
            if len(test_data) % args.checkpoint_every == 0:
                save_checkpoint(test_checkpoint, test_data[-args.checkpoint_every:])

    log.info(f"[main] Test set: {len(test_data)} examples")

    # Save final outputs
    def save_jsonl(filename, data):
        fpath = out_path / filename
        with open(fpath, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        log.info(f"[main] Saved {len(data)} examples to {fpath}")

    save_jsonl("train.jsonl", train_data)
    save_jsonl("val.jsonl", val_data)
    save_jsonl("test.jsonl", test_data)

    stats = {
        "train": len(train_data),
        "val": len(val_data),
        "test": len(test_data),
        "total": len(train_data) + len(val_data) + len(test_data)
    }
    with open(out_path / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    log.info(f"[main] Dataset construction complete. Stats: {stats}")


if __name__ == "__main__":
    main()
