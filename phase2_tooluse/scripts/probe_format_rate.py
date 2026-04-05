"""
筛选格式不稳定的题目：对 GSM8K 前 200 题各采样 4 次，
筛出格式正确率在 20%-80% 之间的题，作为评估集。

支持断点续跑：每题结果实时写入 probe_format_progress.json，
重启时自动从上次位置继续。

格式正确率定义：json.loads 成功 + steps 是 list + answer 是数字
单次生成超时：180 秒，超时算失败（fmt=False）
"""
import re
import json
import os
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_ID = os.path.expanduser("~/.cache/modelscope/hub/models/Qwen/Qwen3___5-0___8B")
GSM8K_DIR = os.path.expanduser("~/.cache/modelscope/datasets/gsm8k/data")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROGRESS_PATH = os.path.join(SCRIPT_DIR, "probe_format_progress.json")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "unstable_format_questions.json")

TIMEOUT_PER_SAMPLE = 180  # 单次生成超时秒数

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

def load_progress():
    """加载已有进度，返回 {index: result} 字典。"""
    if not os.path.exists(PROGRESS_PATH):
        return {}
    with open(PROGRESS_PATH) as f:
        data = json.load(f)
    return {r["index"]: r for r in data}

def save_progress(results: list):
    """保存当前所有结果到进度文件。"""
    with open(PROGRESS_PATH, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

class TimeoutStoppingCriteria(StoppingCriteria):
    """每个 token 生成后检查是否超时，超时则强制停止。"""
    def __init__(self, timeout: float):
        self.deadline = time.time() + timeout
        self.timed_out = False

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if time.time() > self.deadline:
            self.timed_out = True
            return True
        return False


def generate_with_timeout(model, tokenizer, inputs, timeout=180):
    """带超时的生成，超时返回空字符串。"""
    stopper = TimeoutStoppingCriteria(timeout)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([stopper]),
        )
    if stopper.timed_out:
        print(f"    [超时，跳过本次采样]")
        return {"output": "", "timed_out": True}
    new_ids = out[0][inputs["input_ids"].shape[1]:]
    return {"output": tokenizer.decode(new_ids, skip_special_tokens=True), "timed_out": False}

def main():
    N_QUESTIONS = 200
    N_SAMPLES = 4
    FMT_LOW = 0.20
    FMT_HIGH = 0.80

    # 加载已有进度
    done = load_progress()
    start_index = max(done.keys()) + 1 if done else 0
    results = [done[i] for i in sorted(done.keys())]
    print(f"已有进度: {len(done)} 题，从第 {start_index + 1} 题继续")

    print("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map=DEVICE, local_files_only=True,
    )
    model.eval()
    print(f"模型加载完成，继续探测第 {start_index+1}-{N_QUESTIONS} 题\n")

    test_raw = load_jsonl(f"{GSM8K_DIR}/test.jsonl")
    t_start = time.time()
    completed = len(done)

    for i, item in enumerate(test_raw[start_index:N_QUESTIONS], start=start_index):
        question = item["question"]
        correct = extract_answer(item["answer"])

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

        fmt_count = 0
        timeout_count = 0
        for s in range(N_SAMPLES):
            print(f"  [题目{i+1} 采样{s+1}/{N_SAMPLES}] 开始 {time.strftime('%H:%M:%S')}", flush=True)
            r = generate_with_timeout(model, tokenizer, inputs, timeout=TIMEOUT_PER_SAMPLE)
            if r["timed_out"]:
                timeout_count += 1
            elif is_format_valid(r["output"]):
                fmt_count += 1

        fmt_rate = fmt_count / N_SAMPLES
        # 超时 >= 2 次，数据不可信，标记 skip
        skipped = timeout_count >= 2
        result = {
            "index": i,
            "question": question,
            "correct_answer": correct,
            "fmt_rate": fmt_rate,
            "fmt_count": fmt_count,
            "timeout_count": timeout_count,
            "skip": skipped,
        }
        results.append(result)
        save_progress(results)  # 每题完成立即写盘

        completed += 1
        elapsed = time.time() - t_start
        remaining = N_QUESTIONS - start_index - completed
        eta = elapsed / completed * remaining if completed > 0 else 0
        to_str = f" (超时{timeout_count}次，SKIP)" if skipped else (f" (超时{timeout_count}次)" if timeout_count else "")
        print(f"[{i+1:3d}/{N_QUESTIONS}] fmt={fmt_rate:.0%} ({fmt_count}/{N_SAMPLES}){to_str}  "
              f"elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m  q={question[:50]}...")

    # 筛选结果（排除 skip 的题）
    valid = [r for r in results if not r.get("skip")]
    skipped = [r for r in results if r.get("skip")]
    unstable = [r for r in valid if 0.20 <= r["fmt_rate"] <= 0.60]  # 25% 和 50%
    always_fail = [r for r in valid if r["fmt_rate"] < 0.20]
    always_ok = [r for r in valid if r["fmt_rate"] > 0.60]

    print(f"\n{'='*60}")
    print(f"总题数: {len(results)}（跳过 {len(skipped)} 题，超时过多）")
    print(f"格式稳定（>60%）:      {len(always_ok)} 题")
    print(f"格式不稳定（25%-50%）: {len(unstable)} 题  → 评估集")
    print(f"格式基本失败（<20%）:  {len(always_fail)} 题")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(unstable, f, ensure_ascii=False, indent=2)
    print(f"\n评估集已保存: {OUTPUT_PATH}  ({len(unstable)} 题)")


if __name__ == "__main__":
    main()
