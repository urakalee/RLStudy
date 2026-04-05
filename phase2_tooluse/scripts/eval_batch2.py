"""
加载已训练的 LoRA checkpoint，对第二批 20 题各跑 4 次，验证第一批结果是否是巧合。
"""
import re
import json
import os
import time
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_ID = os.path.expanduser("~/.cache/modelscope/hub/models/Qwen/Qwen3___5-0___8B")
CHECKPOINT = "/Users/liqiang/claudelab/rl_study/checkpoints/grpo_unstable_fmt/checkpoint-200"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UNSTABLE_PATH = os.path.join(SCRIPT_DIR, "unstable_format_questions.json")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "train_eval_results.json")

SYSTEM_PROMPT = """你是一个数学解题助手。解题后只输出 JSON，格式如下：

{"steps": ["步骤1", "步骤2", ...], "answer": 数字}

规则：
- steps 是字符串数组，每个元素是一个计算步骤
- answer 是最终答案，必须是数字（不带单位）
- 禁止输出 JSON 以外的任何内容

示例：
问题：A store has 48 apples, sells 15, then receives 30 more. How many now?
{"steps": ["48 - 15 = 33", "33 + 30 = 63"], "answer": 63}
"""

def is_format_valid(output: str) -> bool:
    try:
        obj = json.loads(output.strip())
        steps = obj.get("steps")
        answer = obj.get("answer")
        if not isinstance(steps, list) or answer is None:
            return False
        float(answer)
        return True
    except:
        return False

def generate(model, tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)

def main():
    random.seed(42)

    # 读取评估集，重建第一批/第二批的分割
    with open(UNSTABLE_PATH) as f:
        unstable = json.load(f)
    shuffled = unstable.copy()
    random.shuffle(shuffled)
    batch1_indices = set(item["index"] for item in shuffled[:20])
    batch2 = shuffled[20:40]  # 第二批取20题
    print(f"第二批: {len(batch2)} 题")

    # 加载模型 + LoRA checkpoint
    print("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map=DEVICE, local_files_only=True,
    )
    checkpoint_path = os.path.abspath(CHECKPOINT)
    print(f"加载 LoRA checkpoint: {checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()
    print("模型加载完成\n")

    # 第二批评估
    print(f"=== 第二批评估 ({len(batch2)} 题 × 4 次) ===\n")
    results = []
    for i, item in enumerate(batch2):
        q = item["question"]
        fmt_count = 0
        for s in range(4):
            print(f"  [题目{i+1}/{len(batch2)} 采样{s+1}/4] {time.strftime('%H:%M:%S')}", flush=True)
            output = generate(model, tokenizer, q)
            if is_format_valid(output):
                fmt_count += 1
        fmt_rate = fmt_count / 4
        baseline = item["fmt_rate"]
        delta = fmt_rate - baseline
        results.append({"index": item["index"], "baseline_fmt": baseline, "trained_fmt": fmt_rate, "delta": delta})
        print(f"  题目{i+1}: baseline={baseline:.0%} -> trained={fmt_rate:.0%} ({delta:+.0%})")

    avg_baseline = sum(r["baseline_fmt"] for r in results) / len(results)
    avg_trained = sum(r["trained_fmt"] for r in results) / len(results)
    avg_delta = avg_trained - avg_baseline
    improved = sum(1 for r in results if r["delta"] > 0)

    print(f"\n  平均 baseline: {avg_baseline:.0%}")
    print(f"  平均 trained:  {avg_trained:.0%}")
    print(f"  平均提升:      {avg_delta:+.0%}")
    print(f"  提升题数:      {improved}/{len(results)}")

    # 读取第一批结果合并
    with open(RESULTS_PATH) as f:
        all_results = json.load(f)
    all_results["batch2"] = {
        "avg_baseline": avg_baseline,
        "avg_trained": avg_trained,
        "avg_delta": avg_delta,
        "improved": improved,
        "total": len(results),
        "details": results,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # 综合结论
    batch1_delta = all_results["batch1"]["avg_delta"]
    print(f"\n{'='*60}")
    print(f"第一批提升: {batch1_delta:+.0%}")
    print(f"第二批提升: {avg_delta:+.0%}")
    print(f"综合提升:   {(batch1_delta + avg_delta) / 2:+.0%}")
    if avg_delta >= 0.10:
        print("✓ 两批均有效，方向确认")
    elif avg_delta >= 0.05:
        print("△ 第二批提升偏小，可能存在噪声")
    else:
        print("✗ 第二批无效，第一批可能是巧合")

if __name__ == "__main__":
    main()
