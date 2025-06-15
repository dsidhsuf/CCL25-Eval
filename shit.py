import json
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
from datasets import Dataset
from tqdm import tqdm

# ✅ 模型配置
model_name = "Langboat/mengzi-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# ✅ 强制使用GPU
if not torch.cuda.is_available():
    raise RuntimeError("必须使用GPU训练，请检查CUDA环境")
device = torch.device("cuda")
model = model.to(device)

# ✅ 加载 JSON 数据
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ✅ 清洗数据（仅保留有 content 和非空 output 的样本）
def clean_data(data):
    return [d for d in data if d.get("content") and d.get("output") and d["output"].strip()]

# ✅ 加载并处理训练数据
all_data = clean_data(load_json("train.json"))

# ✅ 自动划分训练集和验证集（9:1）
train_data, val_data = train_test_split(all_data, test_size=0.1, random_state=42)

# ✅ 加载测试集
test_data = load_json("test1.json")

# ✅ 转换为 Huggingface Datasets
train_ds = Dataset.from_list(train_data)
val_ds = Dataset.from_list(val_data)
test_ds = Dataset.from_list(test_data)

# ✅ 分词函数
def tokenize_function(example):
    model_inputs = tokenizer(
        example["content"],
        max_length=256,
        padding="max_length",
        truncation=True,
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["output"],
            max_length=256,
            padding="max_length",
            truncation=True,
        )
    model_inputs["labels"] = [
        (token if token != tokenizer.pad_token_id else -100)
        for token in labels["input_ids"]
    ]
    return model_inputs

# ✅ 分词映射
tokenized_train = train_ds.map(tokenize_function)
tokenized_val = val_ds.map(tokenize_function)

# ✅ 数据打包器
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# ✅ 训练参数
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=100,
    logging_steps=50,
    save_steps=100,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=8,
    weight_decay=0.01,
    save_total_limit=2,
    predict_with_generate=True,
    fp16=True,  # ✅ 启用半精度训练
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# ✅ 初始化 Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# ✅ 开始训练
trainer.train()

# ✅ 保存模型
model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")

# ==========================================
# ✅ 推理测试集并保存结果到 demo.txt
# ==========================================

print("🔎 开始推理测试集...")

model.eval()
with open("demo.txt", "w", encoding="utf-8") as f:
    for item in tqdm(test_data):
        input_text = item["content"]
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=256,
            truncation=True,
            padding="max_length"
        ).to(device)

        output_ids = model.generate(
            **inputs,
            max_length=256,
            num_beams=5,
            early_stopping=True
        )
        pred = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        # ✅ 自动补全 [END]
        if not pred.endswith("[END]"):
            pred += " [END]"

        f.write(pred + "\n")
