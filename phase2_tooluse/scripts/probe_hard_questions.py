"""
方向 D 验证：用 GSM8K 后半段（harder questions）探测 0.8B baseline 格式正确率。
目标：找到一个 baseline 格式正确率在 30-50% 的题目区间，作为 GRPO 训练的目标集。
"""
import re
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_ID = os.path.expanduser("~/.cache/modelscope/hub/models/Qwen/Qwen3___5-0___8B")
GSM8K_DIR = os.path.expanduser("~/.cache/modelscope/datasets/gsm8k/data")

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

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def extract_answer(s):
    m = re.search(r"####\s*([\d,]+)", s)
    return m.group(1).replace(",", "") if m else None

def compute_reward(output: str) -> dict:
    text = output.strip()
    score = 0.0
    has_braces = '{' in text and '}' in text
    if has_braces:
        score += 0.1
    obj = None
    try:
        obj = json.loads(text)
        score += 0.4
    except:
        pass
    fields_ok = False
    if obj is not None:
        steps = obj.get('steps')
        answer = obj.get('answer')
        if isinstance(steps, list) and answer is not None:
            try:
                float(answer)
                fields_ok = True
                score += 0.5
            except:
                pass
    return {
        "braces": 0.1 if has_braces else 0.0,
        "json":   0.4 if obj is not None else 0.0,
        "fields": 0.5 if fields_ok else 0.0,
        "total":  round(score, 2),
    }

def probe_range(model, tokenizer, questions, label, n=10, n_samples=3):
    """对 n 道题各采样 n_samples 次，统计平均格式正确率。"""
    print(f"\n=== {label} (n={n}, samples_per_q={n_samples}) ===")
    all_rewards = []
    perfect_counts = 0
    total_counts = 0

    for i, q in enumerate(questions[:n]):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

        q_rewards = []
        for _ in range(n_samples):
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_ids = out[0][inputs["input_ids"].shape[1]:]
            output = tokenizer.decode(new_ids, skip_special_tokens=True)
            r = compute_reward(output)
            q_rewards.append(r["total"])
            total_counts += 1
            if r["total"] == 1.0:
                perfect_counts += 1

        avg_q = sum(q_rewards) / len(q_rewards)
        all_rewards.append(avg_q)
        print(f"  题目 {i+1:2d}: avg_reward={avg_q:.2f}  samples={q_rewards}  q={q[:60]}...")

    avg = sum(all_rewards) / len(all_rewards)
    fmt_rate = perfect_counts / total_counts
    print(f"\n  汇总: 平均reward={avg:.3f}  完整格式率={fmt_rate:.0%}  ({perfect_counts}/{total_counts})")
    return avg, fmt_rate


def main():
    print("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map=DEVICE,
        local_files_only=True,
    )
    model.eval()
    print("模型加载完成")

    test_raw = load_jsonl(f"{GSM8K_DIR}/test.jsonl")
    total = len(test_raw)
    print(f"测试集总量: {total} 条")

    # 取三个区间各 10 题
    ranges = [
        ("前10题 (index 0-9)",    [d["question"] for d in test_raw[0:10]]),
        ("中段10题 (index 600-609)", [d["question"] for d in test_raw[600:610]]),
        ("后10题 (index 1309-1318)", [d["question"] for d in test_raw[-10:]]),
    ]

    results = {}
    for label, questions in ranges:
        avg, fmt_rate = probe_range(model, tokenizer, questions, label, n=10, n_samples=3)
        results[label] = {"avg_reward": avg, "fmt_rate": fmt_rate}

    print("\n\n========= 汇总对比 =========")
    for label, r in results.items():
        print(f"  {label}: avg_reward={r['avg_reward']:.3f}  格式正确率={r['fmt_rate']:.0%}")


if __name__ == "__main__":
    main()
