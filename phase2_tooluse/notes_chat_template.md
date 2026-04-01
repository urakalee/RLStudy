# Chat Template 笔记

## apply_chat_template 是什么

`apply_chat_template` 是 **transformers 通用方法**，所有 HuggingFace 模型都有。
但**模板内容是模型特有的**——每个模型在发布时会在 `tokenizer_config.json` 里内置一个 Jinja2 模板。

```
apply_chat_template(messages)
    ↓
读取 tokenizer_config.json 里的 chat_template 字段（Jinja2 字符串）
    ↓
用 Jinja2 渲染成该模型的原生格式
```

不同模型渲染出的格式不同：

| 模型 | 格式特征 |
|---|---|
| Qwen3.5 | `<\|im_start\|>system\n...<\|im_end\|>` |
| LLaMA3 | `<\|begin_of_text\|><\|start_header_id\|>system<\|end_header_id\|>...` |
| Mistral | `[INST] ... [/INST]` |

## 主要参数

| 参数 | 类型 | 说明 |
|---|---|---|
| `conversation` | list | 消息列表，必填 |
| `tools` | list | 工具定义（JSON Schema），模型支持时自动注入 system |
| `tokenize` | bool | `False` 返回字符串，`True` 返回 token ids，默认 `True` |
| `add_generation_prompt` | bool | 末尾加 assistant 开头 token，推理时必须为 `True` |
| `return_tensors` | str | `"pt"` / `"tf"`，`tokenize=True` 时生效 |
| `return_dict` | bool | 返回带 `attention_mask` 的字典，默认 `False` |
| `padding` | bool | 批量时是否 padding |
| `truncation` | bool | 超长时是否截断 |
| `max_length` | int | 最大长度，配合 `truncation` 使用 |
| `chat_template` | str | 覆盖模型自带模板，传入自定义 Jinja2 字符串 |
| `documents` | list | RAG 场景，把检索到的文档注入 prompt（部分模型支持） |

最常用：前四个。

## Jinja2 是什么

Jinja2 是 Python 的模板引擎，在普通文本里嵌入逻辑：

```jinja2
{# 注释 #}
{{ 变量 }}
{% if condition %}...{% endif %}
{% for item in list %}...{% endfor %}
```

chat_template 本质就是一段 Jinja2 字符串，`apply_chat_template` 调用时用 Jinja2 引擎渲染它。

## Qwen3.5-2B chat_template 关键逻辑

### 1. tools 注入

```jinja2
{%- if tools and tools is iterable %}
    {{- '<|im_start|>system\n# Tools\n...' }}
    {%- for tool in tools %}
        {{- tool | tojson }}
    {%- endfor %}
    {{- '...调用格式说明...' }}
```

传入 `tools=` 参数后，system prompt 会自动出现工具定义 + 调用格式说明。
这就是原生 function calling 的实现方式——不需要手写 system prompt。

工具调用格式（模板里硬编码）：
```
<tool_call>
<function=函数名>
<parameter=参数名>
参数值
</parameter>
</function>
</tool_call>
```

### 2. think 标签处理

```jinja2
{%- if '</think>' in content %}
    {%- set reasoning_content = content.split('</think>')[0]... %}
    {%- set content = content.split('</think>')[-1]... %}
```

模板会自动把 `<think>...</think>` 从 content 里剥离出来单独处理（reasoning_content）。

### 3. add_generation_prompt 的默认行为

```jinja2
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is true %}
        {{- '<think>\n' }}                {# 开启思考模式，模型继续写 think #}
    {%- else %}
        {{- '<think>\n\n</think>\n\n' }}  {# 默认：空 think，跳过思考 #}
    {%- endif %}
```

**这是 03 实验里 think 始终为空的真正原因**：
默认情况下 `add_generation_prompt=True` 会在末尾追加 `<think>\n\n</think>\n\n`，
相当于告诉模型 think 已经结束了，所以不会再输出 think 内容。

要让模型输出 think，需要传 `enable_thinking=True`：
```python
tokenizer.apply_chat_template(messages, enable_thinking=True, ...)
```
这样末尾只有 `<think>\n`，模型会继续写 think 内容。

## 原生 function calling 验证结论（04 实验）

Qwen3.5-2B 未经任何训练，已能正确做出工具调用决策：
- 简单题（3+5）→ 直接答，不调工具
- 大数乘法（1847×293）→ 主动调 calculator，格式完全正确

说明模型在 SFT 阶段已经学会了这个决策策略，针对计算场景做 RL 训练没有意义。
