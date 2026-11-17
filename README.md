# VLM Fine-tuning

Qwen2.5-VL-3B 파인튜닝 — 음식 사진 + 상황 설명 → 마케팅 JSON(영문 이미지 프롬프트 + 광고 카피 + 전략) 생성.


# 📁 구조

```
VLM/
├── data/        # (data_list.jsonl - 드라이브)
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
# 출력 스키마

```json
{
  "sd_prompt": "string (영문, 이미지 생성 프롬프트)",
  "negative_prompt": "string (영문)",
  "ad_copy": "string (한국어, 짧은 광고 카피)",
  "strategy_reason": "string (한국어, 전략 설명)",
  "aspect_ratio": "1:1 | 4:5 | 9:16"
}
```

+)
| 필드 | 설명 |
|---|---|
| `sd_prompt` | 이미지 생성용 영문 프롬프트. 필수 키워드 `realistic soft drop shadow under the food, natural directional lighting` 포함 |
| `negative_prompt` | 피할 요소 (기본: `text, words, watermark, ...`) |
| `ad_copy` | 이미지 위에 **별도로 얹을** 한국어 카피 (이미지 안에 텍스트 생성 X) |
| `strategy_reason` | 전략 의도 설명 |
| `aspect_ratio` | `1:1` 피드 / `4:5` 세로 피드 / `9:16` 스토리 |



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
