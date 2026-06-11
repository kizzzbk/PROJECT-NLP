# 🏥 BrandHealth AI Pipeline

**Hệ thống Giám sát Sức khỏe Thương hiệu từ Phản hồi Khách hàng ứng dụng Học sâu**

> Phân loại sắc thái cảm xúc (Sentiment Analysis) cho tiếng Việt sử dụng BiLSTM, BiLSTM + Attention, và PhoBERT Fine-tuned, kèm Web Dashboard tương tác.

---

## 📋 Tổng quan

| Tính năng | Mô tả |
|-----------|-------|
| **3 Mô hình AI** | BiLSTM (Baseline), BiLSTM + Attention, PhoBERT Fine-tuned |
| **Tiền xử lý tiếng Việt** | Xử lý teencode, tách từ ghép, chuẩn hóa unicode |
| **Web Dashboard** | Phân tích đơn lẻ + Hàng loạt với biểu đồ tương tác |
| **Experiment Tracking** | Theo dõi từng lần train, so sánh mô hình |
| **Attention Heatmap** | Trực quan hóa từ khóa mô hình tập trung |

## 🚀 Bắt đầu nhanh

### 1. Cài đặt

```bash
# Clone project
git clone <repo-url>
cd "BTL NLP"

# Cài đặt dependencies
pip install -r requirements.txt

# Hoặc dùng Makefile
make setup
```

### 2. Chuẩn bị dữ liệu

```bash
# Nếu có dataset NTC-SCV, đặt vào data/raw/ntc_scv.csv
# Nếu chưa có, script sẽ tạo sample data
python scripts/prepare_data.py
```

### 3. Huấn luyện mô hình

```bash
# Train BiLSTM (nhanh nhất, ~2-5 phút trên CPU)
python scripts/train.py --model bilstm

# Train BiLSTM + Attention
python scripts/train.py --model bilstm_attention

# Train PhoBERT (cần GPU, ~10-30 phút)
python scripts/train.py --model phobert

# Train tất cả
make train-all

# Override hyperparameters qua CLI
python scripts/train.py --model bilstm --epochs 50 --lr 0.0005
```

### 4. Chạy Dashboard

```bash
streamlit run app/app.py
# Hoặc
make serve
```

### 5. Dự đoán từ CLI

```bash
python scripts/predict.py --model bilstm --text "sản phẩm quá tệ, ship chậm"
```

---

## 📁 Cấu trúc dự án

```
BTL NLP/
├── configs/                  # ⚙️ YAML configurations
│   ├── base.yaml             # Shared defaults
│   ├── preprocessing.yaml    # Text cleaning config
│   ├── model_bilstm.yaml     # BiLSTM hyperparameters
│   ├── model_bilstm_attention.yaml
│   ├── model_phobert.yaml    # PhoBERT fine-tuning config
│   └── dashboard.yaml        # Streamlit settings
│
├── src/                      # 🧠 Core source code
│   ├── data/                 # Data loading & preprocessing
│   ├── models/               # Model architectures
│   ├── training/             # Training & evaluation
│   ├── experiment/           # Experiment tracking
│   ├── inference/            # Prediction engine
│   └── utils/                # Config, logging, seed, device
│
├── scripts/                  # 🚀 Entry-point scripts
│   ├── train.py              # Train any model
│   ├── evaluate.py           # Evaluate on test set
│   ├── predict.py            # CLI prediction
│   └── prepare_data.py       # Data preparation
│
├── app/                      # 🌐 Streamlit Web Dashboard
│   ├── app.py                # Main entry point
│   ├── pages/                # Page components
│   └── components/           # Reusable UI components
│
├── experiments/              # 📊 Training run logs
├── data/                     # 📂 Data storage
├── models/                   # 💾 Saved checkpoints
├── tests/                    # 🧪 Unit tests
└── notebooks/                # 📓 Research notebooks
```

---

## 🔬 Kiến trúc Mô hình

### BiLSTM (Baseline)
```
Embedding → BiLSTM (2 layers) → Max+Mean Pooling → FC → Output
```
- Target: F1 ~70-75%
- Fastest training, good baseline

### BiLSTM + Attention
```
Embedding → BiLSTM → Additive Attention → Context Vector → FC → Output
```
- Target: F1 ~75-80%
- Exports attention weights for Heatmap visualization

### PhoBERT Fine-tuned
```
PhoBERT Encoder → [CLS] token → Dropout → Linear → Output
```
- Target: F1 ~85-90%
- Pre-trained Vietnamese language model
- Requires word segmentation (VnCoreNLP/underthesea)

---

## 📈 Experiment Tracking

Mỗi lần train tạo ra folder riêng trong `experiments/`:

```
experiments/
├── 2026-06-09_14-30-00_bilstm_run/
│   ├── config.yaml          # Frozen config snapshot
│   ├── hyperparams.json     # All hyperparameters
│   ├── metrics.json         # Per-epoch metrics
│   ├── training_log.csv     # Per-step granular log
│   ├── system_info.json     # Hardware info
│   ├── summary.json         # Final results
│   └── plots/
│       ├── training_curves.png
│       └── confusion_matrix.png
```

So sánh runs:
```python
from src.experiment.run_manager import RunManager

manager = RunManager('experiments')
df = manager.list_runs()
comparison = manager.compare_runs(['run_1', 'run_2'])
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src --cov-report=html
```

---

## 📄 Configuration

Tất cả hyperparameters được quản lý qua YAML files trong `configs/`.
Priority: `base.yaml` < `model_*.yaml` < CLI overrides.

```bash
# Override via CLI
python scripts/train.py --model bilstm --epochs 50 --lr 0.0005 --batch_size 128
```

---

## 🔧 Makefile Commands

| Command | Description |
|---------|-------------|
| `make setup` | Install dependencies |
| `make prepare-data` | Prepare dataset |
| `make train MODEL=bilstm` | Train specific model |
| `make train-all` | Train all models |
| `make evaluate MODEL=bilstm` | Evaluate model |
| `make serve` | Start dashboard |
| `make test` | Run tests |
| `make clean` | Clean cache files |

---

## 📜 License

Academic project — Phan Anh © 2026
