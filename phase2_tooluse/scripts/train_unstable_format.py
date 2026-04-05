"""
用筛选出的格式不稳定题目做 GRPO 训练。
依赖：scripts/unstable_format_questions.json（由 probe_format_rate.py 生成）

训练目标：提升格式不稳定题的格式正确率
Reward：三层独立（braces +0.1, json.loads +0.4, fields +0.5）
"""
import re
import json
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_ID = os.path.expanduser("~/.cache/modelscope/hub/models/Qwen/Qwen3___5-0___8B")
SCRIPT_DIR = os.path.dirname(__file__)
UNSTABLE_PATH = os.path.join(SCRIPT_DIR, "unstable_format_questions.json")

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

# ── Reward ────────────────────────────────────────────────────────────────────
def compute_reward(output: str) -> float:
    text = output.strip()
    score = 0.0
    if '{' in text and '}' in text:
        score += 0.1
    obj = None
    try:
        obj = json.loads(text)
        score += 0.4
    except:
        pass
    if obj is not None:
        steps = obj.get('steps')
        answer = obj.get('answer')
        if isinstance(steps, list) and answer is not None:
            try:
                float(answer)
                score += 0.5
            except:
                pass
    return round(score, 2)

def _completion_to_str(c) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        for msg in reversed(c):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "")
    return str(c)

def reward_fn(completions, **kwargs):
    return [compute_reward(_completion_to_str(c)) for c in completions]

# ── 评估 ──────────────────────────────────────────────────────────────────────
def run_eval(label, model, tokenizer, samples):
    model.eval()
    ok = 0
    for q, _ in samples:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=512, temperature=0.8,
                do_sample=True, pad_token_id=tokenizer.eos_token_id,
            )
        new_ids = out[0][inputs["input_ids"].shape[1]:]
        output = tokenizer.decode(new_ids, skip_special_tokens=True)
        if compute_reward(output) == 1.0:
            ok += 1
    rate = ok / len(samples)
    print(f"[{label}] 格式完整率: {ok}/{len(samples)} = {rate:.0%}")
    return rate

# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    # 读取筛选结果
    print(f"读取筛选题目: {UNSTABLE_PATH}")
    with open(UNSTABLE_PATH) as f:
        unstable = json.load(f)
    print(f"格式不稳定题共 {len(unstable)} 道")

    # 构建数据集（重复填充到至少 500 条，保证训练够用）
    records = []
    while len(records) < 500:
        for item in unstable:
            records.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": item["question"]},
                ],
                "answer": str(item["correct_answer"]),
            })
            if len(records) >= 500:
                break
    train_dataset = Dataset.from_list(records)
    print(f"训练集: {len(train_dataset)} 条（从 {len(unstable)} 题重复填充）")

    # 评估集：取前 20 道不稳定题（每题单次评估）
    eval_samples = [(item["question"], item["correct_answer"]) for item in unstable[:20]]

    # 加载模型
    print("\n加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map=DEVICE, local_files_only=True,
    )
    model.gradient_checkpointing_enable()

    # LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Baseline 评估
    print("\n--- Baseline ---")
    baseline_rate = run_eval("baseline", model, tokenizer, eval_samples)

    # GRPO 配置
    training_args = GRPOConfig(
        output_dir="./checkpoints/grpo_unstable_fmt",
        num_generations=4,
        max_completion_length=512,
        temperature=0.8,
        beta=0.04,
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=200,
        warmup_steps=20,
        optim="adafactor",
        logging_steps=10,
        save_steps=200,
        report_to="none",
        remove_unused_columns=False,
        seed=42,
    )

    class RewardLogCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            r = (logs.get("rewards/reward_fn/mean")
                 or logs.get("rewards/mean")
                 or logs.get("reward"))
            if r is None:
                return
            print(f"  step {state.global_step:4d} | loss={logs.get('loss', float('nan')):.4f} | reward_mean={r:.4f}")

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        callbacks=[RewardLogCallback()],
    )

    print("\n开始训练...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\n训练完成，耗时 {elapsed/60:.1f} 分钟")

    # 训练后评估
    print("\n--- 训练后 ---")
    trained_rate = run_eval("trained", model, tokenizer, eval_samples)

    print(f"\n========= 结论 =========")
    delta = trained_rate - baseline_rate
    print(f"baseline 格式完整率:  {baseline_rate:.0%}")
    print(f"训练后格式完整率:     {trained_rate:.0%}")
    print(f"提升:                 {delta:+.0%}")
    if delta >= 0.10:
        print("✓ 方向有效：格式正确率显著提升")
    elif delta >= 0.05:
        print("△ 方向部分有效：有提升但幅度小")
    else:
        print("✗ 方向无效：无明显提升")

    # 保存 reward 曲线数据
    log_history = trainer.state.log_history
    curve = []
    for entry in log_history:
        r = (entry.get("rewards/reward_fn/mean")
             or entry.get("rewards/mean")
             or entry.get("reward"))
        if r is not None and "step" in entry:
            curve.append({"step": entry["step"], "reward": r})
    curve_path = os.path.join(SCRIPT_DIR, "train_unstable_fmt_curve.json")
    with open(curve_path, "w") as f:
        json.dump({"baseline": baseline_rate, "trained": trained_rate, "curve": curve}, f, indent=2)
    print(f"reward 曲线已保存: {curve_path}")


if __name__ == "__main__":
    main()
