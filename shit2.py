import json
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback
)
from datasets import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

def parse_quadruples(text):
    """
    将形如 "(A1, B1, C1, D1); (A2, B2, C2, D2)" 的文本
    转为 {("A1","B1","C1","D1"), ("A2","B2","C2","D2")}
    """
    quads = set()
    text = text.strip()
    if not text:
        return quads
    for part in text.split(";"):
        part = part.strip()
        if part.startswith("(") and part.endswith(")"):
            items = [x.strip() for x in part[1:-1].split(",")]
            if len(items) == 4:
                quads.add(tuple(items))
    return quads

def compute_quad_f1(preds, labels):
    """
    计算四元组级别的精确率、召回率和 F1
    preds, labels: list of 解码后字符串
    """
    assert len(preds) == len(labels)
    TP = FP = FN = 0
    for p_str, l_str in zip(preds, labels):
        p_set = parse_quadruples(p_str)
        l_set = parse_quadruples(l_str)
        TP += len(p_set & l_set)
        FP += len(p_set - l_set)
        FN += len(l_set - p_set)
    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}

# 模型配置
model_name = "Langboat/mengzi-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 强制 GPU
if not torch.cuda.is_available():
    raise RuntimeError("使用GPU训练，请检查CUDA环境")
device = torch.device("cuda")
model = model.to(device)

# 加载 JSON 数据
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# 清洗数据
def clean_data(data):
    return [d for d in data if d.get("content") and d.get("output") and d["output"].strip()]

# 处理训练/验证/测试数据
all_data = clean_data(load_json("train.json"))
train_data, val_data = train_test_split(all_data, test_size=0.1, random_state=42)
test_data = load_json("test1.json")

train_ds = Dataset.from_list(train_data)
val_ds   = Dataset.from_list(val_data)
test_ds  = Dataset.from_list(test_data)

# 分词函数
def tokenize_function(example):
    inputs = tokenizer(example["content"],
                       max_length=256, padding="max_length", truncation=True)
    with tokenizer.as_target_tokenizer():
        labs = tokenizer(example["output"],
                         max_length=256, padding="max_length", truncation=True)
    inputs["labels"] = [(t if t != tokenizer.pad_token_id else -100)
                        for t in labs["input_ids"]]
    return inputs

tokenized_train = train_ds.map(tokenize_function, batched=True)
tokenized_val   = val_ds.map(tokenize_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# 日志存储
train_loss_list = []
eval_loss_list  = []
f1_score_list   = []

# compute_metrics
def compute_f1(eval_preds):
    preds, labels = eval_preds

    # 如果 preds 是 tuple，取第0项
    if isinstance(preds, tuple):
        preds = preds[0]

    # 将所有小于 0 的 ID（-100等）都替换成 pad_token_id
    preds  = np.where(preds  >= 0, preds,  tokenizer.pad_token_id)
    labels = np.where(labels >= 0, labels, tokenizer.pad_token_id)

    # 解码
    decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds  = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]

    res = compute_quad_f1(decoded_preds, decoded_labels)
    return {"precision": res["precision"], "recall": res["recall"], "f1": res["f1"]}


# 自定义 Callback 记录 loss & F1
class LogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            train_loss_list.append(logs["loss"])
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            eval_loss_list.append(metrics.get("eval_loss"))
            f1_score_list.append(metrics.get("eval_f1", 0.0))

# 训练参数（确保 save_steps 与 eval_steps 整数倍关系）
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=100,
    logging_steps=10,
    save_steps=100,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    save_total_limit=2,
    predict_with_generate=True,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_f1,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3), LogCallback()]
)

# 开始训练
trainer.train()

# 保存模型
model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")

# 绘制曲线图
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_loss_list, label="Train Loss")
plt.plot(range(0,len(train_loss_list),1), eval_loss_list, label="Eval Loss")
plt.xlabel("Logging Steps")
plt.ylabel("Loss")
plt.legend()
plt.subplot(1,2,2)
plt.plot(range(0,len(f1_score_list)), f1_score_list, label="Eval F1")
plt.xlabel("Evaluation Steps")
plt.ylabel("F1 Score")
plt.legend()
plt.tight_layout()
plt.savefig("training_metrics.png")
plt.show()

# 推理并写入 demo.txt
model.eval()
with open("demo.txt", "w", encoding="utf-8") as f:
    for item in tqdm(test_data):
        inputs = tokenizer(item["content"],
                           return_tensors="pt",
                           max_length=256,
                           truncation=True,
                           padding="max_length").to(device)
        outs = model.generate(**inputs,
                              max_length=256,
                              num_beams=5,
                              early_stopping=True)
        pred = tokenizer.decode(outs[0], skip_special_tokens=True).strip()
        if not pred.endswith("[END]"):
            pred += " [END]"
        f.write(pred + "\n")
