# NanoReason-3B: Step-Aware LoRA for Verifiable Chain-of-Thought Reasoning

> **Nghiên cứu Khoa học Sinh viên** — Trường Đại học Giao thông Vận tải TP. Hồ Chí Minh  
> Chủ nhiệm đề tài: **Phan Đức Tài** · Thành viên: **Trịnh Trần Trung Đức**  
> Thời gian thực hiện: 01/2026 – 03/2026

---

## Tổng quan

**NanoReason-3B** là một mô hình ngôn ngữ nhỏ (Small Language Model) 3 tỷ tham số được tinh chỉnh bằng phương pháp **Step-Aware LoRA** — một kiến trúc tinh chỉnh có giám sát phân cấp mới, nhằm chuyển giao năng lực suy luận toán học có khả năng xác thực (Verifiable Chain-of-Thought) từ mô hình lớn sang mô hình nhỏ.

Thay vì áp dụng Supervised Fine-Tuning truyền thống với hàm mất mát đồng nhất, Step-Aware LoRA ép buộc mô hình phân tách quá trình giải toán thành **4 giai đoạn nhận thức độc lập**:

```
<UNDERSTAND> → <PLAN> → <EXECUTE> → <VERIFY>
```

Mỗi giai đoạn được giám sát riêng với trọng số gradient bất đối xứng, tạo ra lực ép học tập tập trung vào các bước quan trọng nhất về mặt toán học.

---

## Kiến trúc đề xuất

### Sáng kiến 1 — Hierarchical Step Supervision

4 giai đoạn nhận thức được thiết kế dựa trên khung Polya (1945):

| Thẻ | Vai trò | Mô tả |
|-----|---------|-------|
| `<UNDERSTAND>` | Phân tích đề | Xác định dữ kiện, ẩn số, điều kiện biên |
| `<PLAN>` | Lập kế hoạch | Lựa chọn công thức, định hướng thuật toán |
| `<EXECUTE>` | Tính toán | Thực hiện từng bước có đánh số |
| `<VERIFY>` | Kiểm chứng | Thay số ngược, xác nhận tính đúng đắn |

### Sáng kiến 2 — Step-Aware Loss Function

Hàm mất mát phân cấp với trọng số bất đối xứng:

```
L_total = α·L_understand + β·L_plan + γ·L_execute + δ·L_verify + ε·L_transition
```

| Tham số | Giá trị | Lý do |
|---------|---------|-------|
| α (UNDERSTAND) | 0.10 | Bước dễ, mô hình học nhanh |
| β (PLAN) | 0.30 | Lỗi lan truyền cao, phạt nặng |
| γ (EXECUTE) | 0.35 | Ảnh hưởng trực tiếp đáp án cuối |
| δ (VERIFY) | 0.25 | Ổn định bước kiểm chứng |

---

## Tập dữ liệu

### Raw Datasets
[![Kaggle](https://img.shields.io/badge/Kaggle-Raw_Datasets-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/trinhduc041/nckh-datasets)

| Dataset | Split | Số mẫu gốc | Sau lọc (train) | Tỷ lệ |
|---------|-------|-----------|-----------------|-------|
| GSM8K | train | 7.473 | 7.431 | 99.4% |
| GSM8K | test | 1.319 | — | — |
| MATH | train | 11.500 | 10.388 | 90.3% |
| MATH | test | 1.000 | — | — |
| VNHSGE | train | 250 | 225 | 90.0% |
| **Tổng train** | | **19.223** | **18.044** | **93.9%** |

### Processed Dataset (4-Step CoT)
[![Kaggle](https://img.shields.io/badge/Kaggle-Math_CoT_4Step-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/trinhduc041/nckh-processed-data)

Định dạng JSON sau pipeline với 8 trường:

```json
{
  "id": "math_5750",
  "question": "Express as a common fraction: ...",
  "original_solution": "...",
  "cot_4steps": {
    "understand": "Domain: Algebra. Given: complex fraction...",
    "plan": "Key formulas: adding fractions (LCD)...",
    "execute": "1. Numerator: 2/5 + 3/4 = 23/20. ...",
    "verify": "Substitution test: (23/20)/(11/18) = 207/110 ✓"
  },
  "model_prediction": "\\frac{207}{110}",
  "ground_truth": "\\frac{207}{110}",
  "is_correct": true,
  "full_response": "..."
}
```

### Data Generation Scripts

| Dataset | Script |
|---------|--------|
| GSM8K | [![Kaggle](https://img.shields.io/badge/Kaggle-Gen_GSM8K-20BEFF?logo=kaggle)](https://www.kaggle.com/code/trinhduc041/nckh-gsm8k) |
| VNHSGE | [![Kaggle](https://img.shields.io/badge/Kaggle-Gen_VNHSGE-20BEFF?logo=kaggle)](https://www.kaggle.com/code/trinhduc041/nckh-vnhsge) |
| MATH | [![Kaggle](https://img.shields.io/badge/Kaggle-Gen_MATH-20BEFF?logo=kaggle)](https://www.kaggle.com/code/trinhduc041/nckh-gendata-math) |

---

## Mô hình

### NanoReason-3B (Model chính)

[![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-NanoReason--3B-yellow)](https://huggingface.co/ductaiphan/NanoReason-3B)
[![Kaggle](https://img.shields.io/badge/Kaggle-NanoReason--3B-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/ductaiphan/nanoreason-3b)

| Thông số | Giá trị |
|----------|---------|
| Base model | Qwen2.5-3B-Instruct |
| Phương pháp | Step-Aware LoRA |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Tham số cập nhật | ~0.8% tổng tham số |
| Hardware | 2× NVIDIA Tesla T4 (Kaggle) |
| Training time | 1 Epoch (~450 steps) |

### Ablation Models

| Model | Dataset | Mục đích |
|-------|---------|----------|
| [![Kaggle](https://img.shields.io/badge/Kaggle-Standard_LoRA-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/thinguynaas/nckh-week6-baseline-standard-lora) Standard LoRA | GSM8K | Baseline không có cấu trúc 4 bước |
| [![Kaggle](https://img.shields.io/badge/Kaggle-Zero_Penalty-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/thinguynaas/nckh-week6-zero-penalty) Zero Penalty | GSM8K | Không có tín hiệu phạt gradient |
| [![Kaggle](https://img.shields.io/badge/Kaggle-No_CoT_4Steps-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/thinguynaas/nckh-week6-no-cot-4-steps) No CoT 4 Steps | GSM8K | CoT tự do, không có thẻ nhận thức |
| [![Kaggle](https://img.shields.io/badge/Kaggle-LoRA_Rank32-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/thinguynaas/nckh-week6-lora-rank32) LoRA Rank16 | GSM8K | So sánh rank adapter |
| [![Kaggle](https://img.shields.io/badge/Kaggle-Uniform_Loss-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/thinguynaas/nckh-week6-uniform-loss) Uniform Loss | GSM8K | α=β=γ=δ, không phân biệt bước |

---

## Code

### Training & Inference

| Notebook | Mô tả |
|----------|-------|
| [![Kaggle](https://img.shields.io/badge/Kaggle-NCKH_Final-20BEFF?logo=kaggle)](https://www.kaggle.com/code/ductaiphan/nckh-final?scriptVersionId=302282506) | Pipeline huấn luyện chính — StepAwareLoRATrainer |
| [![Kaggle](https://img.shields.io/badge/Kaggle-NanoReason_Preview-20BEFF?logo=kaggle)](https://www.kaggle.com/code/trinhduc041/nanoreason-preview) | Demo inference NanoReason-3B |

### Evaluation

| Notebook | Mô tả |
|----------|-------|
| [![Kaggle](https://img.shields.io/badge/Kaggle-RFS_SVR_Eval-20BEFF?logo=kaggle)](https://www.kaggle.com/code/trinhduc041/rfs-svr-evaluation) | Tính RFS (Reasoning Format Score) và SVR (Step Verification Rate) |

### Ablation Studies

| Notebook | Kịch bản |
|----------|---------|
| [![Kaggle](https://img.shields.io/badge/Kaggle-Ablation_Study_1-20BEFF?logo=kaggle)](https://www.kaggle.com/code/trinhduc041/nckh-week6-ablation-study-1) | Ablation Study phần 1 |
| [![Kaggle](https://img.shields.io/badge/Kaggle-Ablation_Study_2-20BEFF?logo=kaggle)](https://www.kaggle.com/code/thinguynaas/nckh-week6-ablation-study-2) | Ablation Study phần 2 |
| [![Kaggle](https://img.shields.io/badge/Kaggle-No_CoT_4Steps-20BEFF?logo=kaggle)](https://www.kaggle.com/code/trinhduc041/nckh-week6-no-cot-4-steps) | Kịch bản không có cấu trúc 4 bước |
| [![Kaggle](https://img.shields.io/badge/Kaggle-LoRA_Rank32-20BEFF?logo=kaggle)](https://www.kaggle.com/code/trinhduc041/nckh-week6-lora-rank32) | Kịch bản LoRA rank khác nhau |
| [![Kaggle](https://img.shields.io/badge/Kaggle-Uniform_Loss-20BEFF?logo=kaggle)](https://www.kaggle.com/code/trinhduc041/nckh-week6-uniform-loss) | Kịch bản Uniform Loss |
| [![Kaggle](https://img.shields.io/badge/Kaggle-Standard_LoRA-20BEFF?logo=kaggle)](https://www.kaggle.com/code/thinguynaas/nckh-week6-standard-lora) | Kịch bản Standard LoRA baseline |
| [![Kaggle](https://img.shields.io/badge/Kaggle-Zero_Penalty-20BEFF?logo=kaggle)](https://www.kaggle.com/code/thinguynaas/nckh-week6-no-epsilon) | Kịch bản Zero Penalty |

---

## Data Pipeline

```
Raw Datasets (GSM8K / MATH / VNHSGE)
          ↓
  Teacher Model: Qwen-2.5-32B-Instruct
  (System Prompt → XML 4 thẻ, temperature=0.3)
          ↓
  Quality Filter (3 lớp nối tiếp):
    1. Syntax Verification  — đủ 4 thẻ đúng thứ tự?
    2. Math Verification    — đáp án khớp ground truth?
    3. Step Consistency     — EXECUTE ↔ PLAN nhất quán?
          ↓
  Parser (Regex + Rule-based)
  → JSON schema 8 trường
  → Binary label matrices (4 masks)
          ↓
  StepAwareDataset (PyTorch)
          ↓
  StepAwareLoRATrainer
  → HierarchicalStepLoss
  → LoRA adapters (rank=16, alpha=32)
          ↓
  NanoReason-3B
```

---

## Kết quả thực nghiệm

### Accuracy trên GSM8K test (1.319 mẫu, Zero-shot)

> Kết quả chi tiết xem tại Bảng 4.1 trong báo cáo.

### Ablation Study — Đóng góp của từng thành phần

| Kịch bản | Total Loss | Nhận xét |
|----------|-----------|---------|
| **NanoReason-3B** *(Ours)* | ~1.75 | Tốt nhất trên cả 4 trục UPEV |
| No CoT 4 Steps | ~1.82 | VERIFY dao động, mất neo logic |
| Uniform Loss | cao hơn | PLAN và EXECUTE yếu rõ rệt |
| Zero Penalty | cao nhất | Học pattern bề mặt, không học trọng tâm |
| LoRA Rank16 + Uniform | ~1.78 | Sát NanoReason, rank không phải yếu tố quyết định |
| Standard LoRA | N/A | Không có per-step accuracy (không có cấu trúc 4 bước) |

### Chỉ số đánh giá mới

- **RFS (Reasoning Format Score)** — Tỷ lệ chuỗi suy luận duy trì đủ cấu trúc 4 bước không bị đứt gãy
- **SVR (Step Verification Rate)** — Tỷ lệ mô hình tự phát hiện và sửa lỗi tại bước VERIFY

---

## Môi trường & Cài đặt

```python
# Dependencies
torch>=2.0
transformers>=4.40
peft>=0.10
datasets
vllm  # for inference

# Hardware
# Training: 2× NVIDIA Tesla T4 (~29GB VRAM) — Kaggle
# Inference: 1× NVIDIA Tesla T4 (16GB VRAM)
```

### Cấu hình huấn luyện chính

```python
FULL_CONFIG = {
    'model_name': 'Qwen/Qwen2.5-3B-Instruct',
    'lora_rank': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.05,
    'learning_rate': 2e-4,
    'num_epochs': 1,
    'batch_size': 4,
    'gradient_accumulation_steps': 8,
    'max_length': 2048,
    'warmup_steps': 50,
    'loss_weights': {
        'alpha': 0.15,   # UNDERSTAND
        'beta':  0.35,   # PLAN
        'gamma': 0.35,   # EXECUTE
        'delta': 0.15,   # VERIFY
        'epsilon': 0.00  # TRANSITION
    }
}
```

---

## Inference

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = PeftModel.from_pretrained(base_model, "ductaiphan/NanoReason-3B")

prompt = """Solve the following math problem step by step.
Problem: Tom has 15 apples. He gives 4 to Sarah and buys 7 more. How many does he have?"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1, top_p=0.95)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Đầu ra mong đợi:
```
<UNDERSTAND>
Tom bắt đầu với 15 quả táo. Cần tìm số táo sau khi cho đi 4 và mua thêm 7.
</UNDERSTAND>
<PLAN>
Bước 1: Tính sau khi cho: 15 - 4. Bước 2: Tính sau khi mua thêm: kết_quả + 7.
</PLAN>
<EXECUTE>
1. 15 - 4 = 11
2. 11 + 7 = 18
</EXECUTE>
<VERIFY>
Kiểm tra: 15 - 4 = 11 ✓, 11 + 7 = 18 ✓. Đáp án: 18.
</VERIFY>
```

---

## Cấu trúc Repository

```
NanoReason-3B/
├── data/
│   ├── raw/                    # GSM8K, MATH, VNHSGE gốc
│   └── processed/              # JSON 4-step sau pipeline
├── src/
│   ├── dataset.py              # StepAwareDataset
│   ├── loss.py                 # HierarchicalStepLoss
│   ├── trainer.py              # StepAwareLoRATrainer
│   └── parser.py               # XML → label matrices
├── notebooks/
│   ├── nckh-final.ipynb        # Training pipeline chính
│   ├── ablation_study_1.ipynb
│   ├── ablation_study_2.ipynb
│   └── rfs_svr_evaluation.ipynb
└── README.md
```

---

## Trích dẫn

Nếu bạn sử dụng NanoReason-3B hoặc phương pháp Step-Aware LoRA trong nghiên cứu của mình, xin vui lòng trích dẫn:

```bibtex
@misc{phan2026nanoreason,
  title   = {Step-Aware LoRA: Knowledge Distillation of Verifiable Chain-of-Thought
             Reasoning into Small Language Models via Hierarchical Step Supervision},
  author  = {Phan, Duc Tai and Trinh, Tran Trung Duc},
  year    = {2026},
  school  = {Ho Chi Minh City University of Transport},
  note    = {Student Research Project}
}
```

---

## Nhóm nghiên cứu

| Thành viên | MSSV | Lớp | Vai trò |
|-----------|------|-----|---------|
| Phan Đức Tài | 0962 04003494 | CN 2304CLCB | Chủ nhiệm đề tài |
| Trịnh Trần Trung Đức | 0962 05009137 | CN 2302CLCB | Thành viên |

Viện Đào tạo Chất lượng cao — Trường ĐH Giao thông Vận tải TP. HCM

---

*Đề tài thực hiện từ tháng 01/2026 đến tháng 03/2026.*
