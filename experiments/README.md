# 📊 Experiment Logs

Thư mục này chứa logs từ mỗi lần huấn luyện mô hình.

## Cấu trúc

Mỗi lần chạy `python scripts/train.py` sẽ tạo ra 1 thư mục con:

```
experiments/
├── <timestamp>_<run_name>/
│   ├── config.yaml          # Config đã dùng (frozen snapshot)
│   ├── hyperparams.json     # Hyperparameters
│   ├── metrics.json         # Metrics theo từng epoch
│   ├── training_log.csv     # Log chi tiết theo từng step
│   ├── system_info.json     # Thông tin hệ thống (GPU, Python, etc.)
│   ├── summary.json         # Tổng kết (thời gian, best metric)
│   └── plots/               # Biểu đồ training curves
│       ├── training_curves.png
│       └── confusion_matrix.png
```

## So sánh Runs

```python
from src.experiment.run_manager import RunManager

manager = RunManager('experiments')

# Liệt kê tất cả runs
print(manager.list_runs())

# So sánh cụ thể
comparison = manager.compare_runs(['run_id_1', 'run_id_2'])
print(comparison.to_markdown())

# Tìm best run
best = manager.get_best_run(metric='best_val_f1')
```

## Naming Convention

Run ID format: `YYYY-MM-DD_HH-MM-SS_<model>_<description>`

Ví dụ:
- `2026-06-09_14-30-00_bilstm_baseline`
- `2026-06-09_16-00-00_bilstm_attention_lr0001`
- `2026-06-09_18-00-00_phobert_finetune_full`
