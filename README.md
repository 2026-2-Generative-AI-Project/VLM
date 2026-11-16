# VLM Fine-tuning

Qwen2.5-VL-3B 파인튜닝 — 음식 사진 + 상황 설명 → 마케팅 JSON(영문 이미지 프롬프트 + 광고 카피 + 전략) 생성.


# 📁 구조

```
VLM/
├── data/data_list.jsonl         # 학습 데이터 240개
├── data_prep/dataset.ipynb      # 데이터셋 구성
├── Qwen/qwen_vl_inference.ipynb # 베이스 모델 추론
└── finetune/
    ├── finetune_colab.ipynb     # Colab (QLoRA 4bit)
    └── finetune_mac.py          # Mac (LoRA)
```

+) `images/`, `qwen_vl_lora/`는 깃 제외 → 드라이브에서 다운로드.


# 각 샘플 포맷 (Qwen2.5-VL messages)

```json
{
  "messages": [
    {"role": "system", "content": "마케팅 전략가 시스템 프롬프트..."},
    {"role": "user", "content": [
      {"type": "image", "image": "images/edge_fail_01.jpg"},
      {"type": "text", "text": "비 와서 손님 없어. 남은 빵 팔고 싶어"}
    ]},
    {"role": "assistant", "content": "{...정답 JSON...}"}
  ]
}
```


# 🚀 실행

```bash
pip install -r requirements.txt   # Colab: + bitsandbytes | Mac: bitsandbytes 제외(미지원)
```

⚠️ 실행 전 경로 수정 필수

각 파일 상단에 있는 경로 변수를 본인 환경에 맞게 수정하세요.
- `IMAGE_DIR` = 이미지 폴더 경로
- `JSONL_PATH` = `data_list.jsonl` 경로


# 🛠️ 환경

| | Colab T4 | Mac M4 Max |
|---|---|---|
| 양자화 | 4bit QLoRA | LoRA (bitsandbytes 미지원) |
| 메모리 | ~8GB | 15~20GB |
