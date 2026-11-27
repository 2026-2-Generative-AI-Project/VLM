## VLM — 요청 해석 (AdCanvas 2단계)

[AdCanvas](https://github.com/2026-2-Generative-AI-Project/AdCanvas)의 2단계입니다. 
누끼 음식 이미지와 사용자의 한국어 입력을 함께 보고, 배경 생성 프롬프트(영어)와 광고 문구(한국어)를 JSON으로 만듭니다.
Qwen2.5-VL-3B-Instruct에 LoRA 파인튜닝을 적용했습니다. 

VLM repository: 학습 데이터 구축 + 파인튜닝 코드
서비스 코드 위치: AdCanvas main ntebook 2번 섹션

```
사진 → ① Segmentation → ② VLM (이 레포) → ③ 배경 생성 → ④ 원본 음식 재합성 → 광고 이미지 + 광고 문구
```

# 구조

```
VLM/
├── data_prep/dataset.ipynb      # 데이터셋 구성
├── Qwen/
    ├── qwen_vl_inference.ipynb # 베이스 모델 추론
    └── qwen_vl_lora_mac/       # Mac (LoRA) 파인튜닝 모델
└── finetune/
    ├── finetune_colab.ipynb     # Colab (QLoRA 4bit)
    └── finetune_mac.py          # Mac (LoRA)
```

- 학습 이미지(`images/`)와 LoRA 어댑터(`qwen_vl_lora_mac/`)는 레포에 올리지 않음 → 공유 드라이브 `AdCanvas_nukki`
- 서비스에는 Mac (LoRA) 어댑터 사용

# input / output

**input**
- 누끼 이미지 → 긴 변 512, 회색 `RGB(128,128,128)` 배경
- 사용자 입력
  ```
  상품명: 제육덮밥
  가게명: 미입력
  광고 요구사항: 점심 특선 할인 이벤트
  대상 고객: 직장인
  원하는 분위기: AI 추천
  광고 비율: 4:5
  ```

**output (JSON)**
| 키 | 내용 |
|---|---|
| `sd_prompt` | 영어 배경 프롬프트 (메인 음식은 묘사 안 함) |
| `negative_prompt` | 생성 모델이 피할 것 |
| `ad_copy` | 한국어 광고 문구 |
| `strategy_reason` | 한국어 전략 설명 |
| `aspect_ratio` | `1:1` / `4:5` / `9:16` |

# 학습 데이터

- 음식 사진 100장, 샘플 240개 (`dataset.jsonl`)
- 예외 포함: 손·인형이 함께 찍힌 사진, 여러 음식, 빈 입력·무의미한 입력
- 가게명은 전부 `미입력`, 대상 고객은 230개가 `미입력`

# 학습 설정 (`finetune_mac.py`)

| 항목 | 값 |
|---|---|
| 베이스 모델 | Qwen2.5-VL-3B-Instruct |
| LoRA | r=16, alpha=32, dropout=0.05, 언어 모델 쪽만 |
| 학습 | 3 epoch, lr 2e-4, 배치 1 × 누적 8 |
| 검증 | 사진 단위 10% |
| loss | 정답 JSON 부분만 |

# 실행

```bash
pip install "transformers>=4.49" accelerate peft qwen-vl-utils pillow
cd finetune
python finetune_mac.py   # 같은 폴더에 dataset.jsonl, images/ 필요
```
- 결과:`qwen_vl_lora_mac`폴더 

# 평가 (베이스 vs 파인튜닝)

`AdCanvas_eval.ipynb` 2번 섹션 · 사진 5장, 기본 10 / adversarial 8 / 상충 5 · 후처리 전 출력 기준

| 지표 | 베이스 | 파인튜닝 |
|---|---|---|
| JSON 파싱 · 필수 키 · 비율 준수 | 100% | 100% |
| 그림자·자연광 문구 포함 | 80% | **100%** |
| 상충 요청 음식 미묘사 | 80% | **100%** |
| 규칙 5개 모두 준수 (기본) | 80% | **90%** (검수 후 100%) |
| 평균 추론 시간 | 9.09초 | **6.66초** |

- 형식은 베이스도 100% → 파인튜닝 효과는 도메인 규칙과 짧은 출력에서 나타남
- 케이스가 적어 차이는 1~2건 수준

# 라이선스
Qwen2.5-VL-3B-Instruct: Qwen Research License (비상업 용도만)