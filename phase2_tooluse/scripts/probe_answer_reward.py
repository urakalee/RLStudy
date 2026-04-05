"""
方向 B 验证：探测答案正确率分布，确认是否有"偶尔对"的梯度信号。

思路：
- 对前 20 题各采样 6 次，统计每题的答案正确率
- 找出"偶尔对"的题（正确率在 10%-90% 之间）——这些题才有 GRPO 梯度信号
- 如果大多数题要么全错、要么全对，reward 加答案分也没有意义
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

def normalize(s):
    if s is None:
        return ""
    s = re.sub(r"[,，]", "", str(s).strip())
    s = re.sub(r"[^0-9.\-]", "", s)
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except:
        return s

def parse_json_answer(output: str):
    """从输出中严格解析 answer 字段。"""
    try:
        obj = json.loads(output.strip())
        return obj.get("answer")
    except:
        return None

def main():
    print("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map=DEVICE, local_files_only=True,
    )
    model.eval()
    print("模型加载完成\n")

    test_raw = load_jsonl(f"{GSM8K_DIR}/test.jsonl")

    N_QUESTIONS = 20
    N_SAMPLES = 6

    fmt_ok = 0       # 格式正确次数
    ans_ok = 0       # 答案正确次数
    total = 0

    sometimes_correct = []   # 偶尔对的题
    always_wrong = []        # 全错的题
    always_right = []        # 全对的题

    print(f"{'题目':>4}  {'正确答案':>8}  {'格式率':>6}  {'答案率':>6}  {'样本答案'}")
    print("-" * 70)

    for i, item in enumerate(test_raw[:N_QUESTIONS]):
        question = item["question"]
        correct = normalize(extract_answer(item["answer"]))

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

        q_fmt = 0
        q_ans = 0
        sample_answers = []

        for _ in range(N_SAMPLES):
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

            # 格式判断
            try:
                obj = json.loads(output.strip())
                steps = obj.get("steps")
                answer = obj.get("answer")
                fmt_valid = isinstance(steps, list) and answer is not None
                try:
                    float(answer)
                except:
                    fmt_valid = False
            except:
                obj = None
                fmt_valid = False
                answer = None

            if fmt_valid:
                q_fmt += 1
                fmt_ok += 1

            # 答案判断（只在格式正确时才判断）
            if fmt_valid and answer is not None:
                pred = normalize(str(answer))
                if pred == correct:
                    q_ans += 1
                    ans_ok += 1
                sample_answers.append(pred)
            else:
                sample_answers.append("✗")

            total += 1

        fmt_rate = q_fmt / N_SAMPLES
        ans_rate = q_ans / N_SAMPLES

        status = ""
        if ans_rate == 0.0:
            always_wrong.append(i + 1)
            status = "全错"
        elif ans_rate == 1.0:
            always_right.append(i + 1)
            status = "全对"
        else:
            sometimes_correct.append((i + 1, ans_rate))
            status = f"偶尔对({ans_rate:.0%})"

        print(f"题目{i+1:2d}  {correct:>8}  {fmt_rate:>6.0%}  {ans_rate:>6.0%}  {sample_answers}  {status}")

    print("\n" + "=" * 70)
    print(f"总格式正确率: {fmt_ok}/{total} = {fmt_ok/total:.0%}")
    print(f"总答案正确率: {ans_ok}/{total} = {ans_ok/total:.0%}")
    print()
    print(f"全对的题（无梯度，reward全1）: {always_right}")
    print(f"全错的题（无梯度，reward全0）: {always_wrong}")
    print(f"偶尔对的题（有梯度信号）:     {[(q, f'{r:.0%}') for q, r in sometimes_correct]}")
    print()
    print(f"有梯度信号的题占比: {len(sometimes_correct)}/{N_QUESTIONS} = {len(sometimes_correct)/N_QUESTIONS:.0%}")
    print()
    print("结论：")
    if len(sometimes_correct) >= 8:
        print("  ✓ 方向 B 可行：超过 40% 的题有答案梯度信号，加答案 reward 值得尝试")
    elif len(sometimes_correct) >= 4:
        print("  △ 方向 B 部分可行：约 20-40% 的题有信号，信号偏弱但可尝试")
    else:
        print("  ✗ 方向 B 不可行：绝大多数题要么全对要么全错，加答案 reward 无效")

if __name__ == "__main__":
    main()
