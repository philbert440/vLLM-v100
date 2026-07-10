# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import hashlib
import json
import time
from collections import Counter

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams

PROMPTS = [
    (
        "qa_http",
        "请用中文详细解释 HTTP/2 和 HTTP/3 的核心区别，覆盖传输层、"
        "多路复用、丢包影响和部署成本。",
    ),
    (
        "coding_lru",
        "请用 Python 3 写一个线程安全的 LRUCache，支持 get、put、"
        "delete、__len__，并给出最小可运行测试。",
    ),
    (
        "reasoning_schedule",
        "A、B、C、D、E 五名工程师排 5 天值班：A 不能周一和周五；"
        "B 必须比 C 早；D 不能紧挨 B；E 只能周二到周四；"
        "如果 C 在周三之后，A 必须在 C 前。给出一个满足条件的排班并简要说明。",
    ),
]


def _max_char_run(text: str) -> int:
    best = cur = 0
    prev = None
    for char in text:
        if char == prev:
            cur += 1
        else:
            cur = 1
            prev = char
        best = max(best, cur)
    return best


def _repeat_ngram_ratio(text: str, n: int = 8) -> float:
    chars = [char for char in text if not char.isspace()]
    if len(chars) < n:
        return 0.0
    grams = ["".join(chars[i : i + n]) for i in range(len(chars) - n + 1)]
    counts = Counter(grams)
    repeated = sum(value - 1 for value in counts.values() if value > 1)
    return repeated / len(grams)


def _punct_ratio(text: str) -> float:
    if not text:
        return 0.0
    punct = set("!！?？。。，,、；;：:\n\t \r")
    return sum(1 for char in text if char in punct) / len(text)


def _looks_corrupt(text: str) -> bool:
    return (
        _punct_ratio(text) > 0.8
        or _max_char_run(text) >= 64
        or _repeat_ngram_ratio(text) >= 0.55
    )


def _build_chat_prompt(tokenizer, prompt: str, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/ymzx/models/Qwen3.6-35B-A3B-FP8")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-output-tokens", type=int, default=384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--attention-backend", default="FLASH_ATTN_V100")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    prompts = [
        _build_chat_prompt(tokenizer, prompt, args.enable_thinking)
        for _, prompt in PROMPTS
    ]

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        quantization="fp8",
        dtype="float16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_auto_trim_ratio=0.0,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        max_num_batched_tokens=args.max_num_batched_tokens,
        skip_mm_profiling=True,
        mm_processor_cache_gb=0,
        limit_mm_per_prompt={"image": 0, "video": 0},
        attention_backend=args.attention_backend,
    )
    llm.generate(
        [prompts[0]],
        SamplingParams(max_tokens=16, temperature=0.0),
        use_tqdm=False,
    )

    sampling_params = SamplingParams(max_tokens=args.max_output_tokens, temperature=0.0)
    results = []
    for (case_id, _), prompt in zip(PROMPTS, prompts):
        start = time.time()
        output = llm.generate([prompt], sampling_params, use_tqdm=False)[0].outputs[0]
        elapsed = time.time() - start
        text = output.text
        results.append(
            {
                "id": case_id,
                "prompt_tokens": len(
                    tokenizer.encode(prompt, add_special_tokens=False)
                ),
                "output_tokens": len(output.token_ids),
                "elapsed_sec": elapsed,
                "output_toks_per_sec": (
                    len(output.token_ids) / elapsed if elapsed else None
                ),
                "max_char_run": _max_char_run(text),
                "repeat_8gram_ratio": _repeat_ngram_ratio(text),
                "punct_ratio": _punct_ratio(text),
                "looks_all_punct": _punct_ratio(text) > 0.8,
                "looks_corrupt": _looks_corrupt(text),
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "text_prefix": text[:900],
                "text_suffix": text[-900:],
            }
        )

    payload = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "max_output_tokens": args.max_output_tokens,
        "enable_thinking": args.enable_thinking,
        "results": results,
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
