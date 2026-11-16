"""
Qwen2.5-VL-3B-Instruct JSON 파인튜닝 (QLoRA)

실행 방법:
    1. 설치: pip install "transformers>=4.49" accelerate peft bitsandbytes qwen-vl-utils openpyxl pillow
    2. 폴더 구조:
        ./finetune.py          ← 이 파일
        ./data_list.py         ← 데이터 파일(240장)
        ./images/              ← 누끼 PNG 100장 (edge_fail_01.PNG ~ food_tteok_03.PNG)
    3. 실행: python finetune.py
    4. 결과: ./qwen_vl_lora/ 폴더에 LoRA 어댑터 저장

+) mac버전 - 4bit 양자화 x

"""

import os, re, json, random
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

# =====================================================
# data_list.jsonl 에서 로드
# =====================================================
JSONL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_list.jsonl")

data_list = []
system_prompt = None
with open(JSONL_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        messages = entry["messages"]

        # system / user / assistant 추출
        sys_msg = next(m for m in messages if m["role"] == "system")
        usr_msg = next(m for m in messages if m["role"] == "user")
        ast_msg = next(m for m in messages if m["role"] == "assistant")

        if system_prompt is None:
            system_prompt = sys_msg["content"]

        # user content에서 image와 text 추출
        image_path, user_text = None, ""
        for c in usr_msg["content"]:
            if c["type"] == "image":
                image_path = c["image"]
            elif c["type"] == "text":
                user_text = c["text"]

        # assistant content는 JSON 문자열
        assistant_json = json.loads(ast_msg["content"])

        data_list.append({
            "image": image_path,
            "user_text": user_text,
            "assistant_json": assistant_json,
        })

print(f"JSONL 로드 완료: {len(data_list)}개 샘플")

# =====================================================
# 1. 경로 & 하이퍼파라미터 (필요시 수정)
# =====================================================
MODEL_ID   = "Qwen/Qwen2.5-VL-3B-Instruct"
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))   # 현재 스크립트 위치
IMAGE_DIR  = os.path.join(BASE_DIR, "images")
OUTPUT_DIR = os.path.join(BASE_DIR, "qwen_vl_lora")

MAX_SIDE   = 512
MAX_PIXELS = 512 * 512
EVAL_RATIO = 0.1
SEED       = 42
EPOCHS     = 3
BATCH_SIZE = 1
GRAD_ACCUM = 8   # 실질 배치 = 8
LR         = 2e-4

# T4는 bfloat16 미지원 → float16 자동 선택
COMPUTE_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 60)
print(f"MODEL: {MODEL_ID}")
print(f"DEVICE: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"DTYPE: {COMPUTE_DTYPE}")
print(f"IMAGE_DIR: {IMAGE_DIR}")
print(f"OUTPUT_DIR: {OUTPUT_DIR}")
print(f"샘플 수: {len(data_list)}")
print("=" * 60)

# =====================================================
# 2. 이미지 전처리 (회색 배경 변환)
# =====================================================
def preprocess_image(path):
    img = Image.open(path)
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        gray_bg = Image.new("RGB", img.size, (128, 128, 128))
        gray_bg.paste(img, mask=img.split()[3])
        return gray_bg
    return img.convert("RGB")

# =====================================================
# 3. data_list → 학습용 샘플 준비
# =====================================================
samples, skipped = [], []
for i, item in enumerate(data_list):
    img_name = os.path.basename(item["image"])
    img_name = img_name.replace(".PNG", ".jpg").replace(".png", ".jpg")
    img_path = os.path.join(IMAGE_DIR, img_name)
    if not os.path.exists(img_path):
        skipped.append((i, img_name))
        continue
    samples.append({
        "image_path": img_path,
        "user_text": item["user_text"],
        "target_json": json.dumps(item["assistant_json"], ensure_ascii=False, indent=2),
    })

print(f"\n유효 샘플: {len(samples)} / 제외: {len(skipped)}")
if skipped:
    print("제외된 이미지 (첫 5개):", [s[1] for s in skipped[:5]])

random.shuffle(samples)
n_eval = max(1, int(len(samples) * EVAL_RATIO)) if len(samples) >= 10 else 0
eval_samples, train_samples = samples[:n_eval], samples[n_eval:]
print(f"train: {len(train_samples)} / eval: {len(eval_samples)}")


class VLMDataset(Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

train_ds = VLMDataset(train_samples)
eval_ds  = VLMDataset(eval_samples) if eval_samples else None
processor = AutoProcessor.from_pretrained(MODEL_ID, max_pixels=MAX_PIXELS)
processor.tokenizer.padding_side = "right"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=COMPUTE_DTYPE,
    device_map="auto",
)
model.config.use_cache = False

# LoRA는 언어모델 쪽에만 (비전 인코더 'visual.*' 제외)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =====================================================
# 5. 배치 collator (정답 JSON에만 loss)
# =====================================================
def build_messages(user_text, target_json=None):
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": user_text},
        ]},
    ]
    if target_json is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": target_json}]})
    return msgs

ASSISTANT_START = processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

def find_last(seq, sub):
    for i in range(len(seq) - len(sub), -1, -1):
        if seq[i:i + len(sub)] == sub:
            return i
    return -1

def collate_fn(batch):
    texts, images = [], []
    for ex in batch:
        images.append(preprocess_image(ex["image_path"]))
        texts.append(processor.apply_chat_template(
            build_messages(ex["user_text"], ex["target_json"]),
            tokenize=False, add_generation_prompt=False,
        ))
    enc = processor(text=texts, images=images, padding=True, return_tensors="pt")

    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    for i in range(labels.size(0)):
        start = find_last(enc["input_ids"][i].tolist(), ASSISTANT_START)
        if start == -1:
            labels[i, :] = -100
        else:
            labels[i, : start + len(ASSISTANT_START)] = -100
    enc["labels"] = labels
    return enc

# 배치 하나 미리 검증
_check = collate_fn([train_samples[0]])
_lbl = _check["labels"][0]
print(f"\n[마스킹 검증] 입력 토큰: {_lbl.numel()}, 학습 대상 토큰: {(_lbl != -100).sum().item()}")

# =====================================================
# 6. 학습 시작
# =====================================================
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    logging_steps=5,
    eval_strategy="epoch" if eval_ds else "no",
    save_strategy="epoch",
    save_total_limit=2,
    fp16=(COMPUTE_DTYPE == torch.float16),
    bf16=(COMPUTE_DTYPE == torch.bfloat16),
    optim="adamw_torch",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collate_fn,
)

print("\n학습 시작!")
trainer.train()

trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"\n학습 완료! 저장 위치: {OUTPUT_DIR}")

# =====================================================
# 7. 추론 테스트 (학습된 어댑터)
# =====================================================
print("\n추론 테스트...")
model.eval()
model.config.use_cache = True

def extract_json(text):
    text = text.replace("```json", "").replace("```", "")
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m: return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None

@torch.no_grad()
def generate(image_path, user_text):
    img = preprocess_image(image_path)
    prompt = processor.apply_chat_template(
        build_messages(user_text), tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[img], return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

test_items = eval_samples[:3] if eval_samples else train_samples[:3]
for ex in test_items:
    raw = generate(ex["image_path"], ex["user_text"])
    parsed = extract_json(raw)
    print("=" * 60)
    print("입력:", ex["user_text"] or "(빈 입력)")
    print("파싱:", "성공" if parsed else "실패")
    if parsed:
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
    else:
        print(raw)
