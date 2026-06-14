import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'font.family': 'sans-serif'
})

exp_dir = r"d:\Download\BTL NLP\experiments"
plots_dir = r"d:\Download\BTL NLP\reports\plots"
os.makedirs(plots_dir, exist_ok=True)

# 1. Gather run metrics
bilstm_runs = sorted(glob.glob(os.path.join(exp_dir, "*_bilstm_run_r*")))
bilstm_attn_runs = sorted(glob.glob(os.path.join(exp_dir, "*_bilstm_attention_run_r*")))

def parse_runs(run_paths):
    all_train_loss = []
    all_val_loss = []
    all_val_f1 = []
    
    for p in run_paths:
        metrics_file = os.path.join(p, "metrics.json")
        if not os.path.exists(metrics_file):
            continue
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        
        all_train_loss.append([m["train_loss"] for m in metrics])
        all_val_loss.append([m["val_loss"] for m in metrics])
        all_val_f1.append([m["val_f1_macro"] for m in metrics])
        
    return all_train_loss, all_val_loss, all_val_f1

bilstm_train_loss, bilstm_val_loss, bilstm_val_f1 = parse_runs(bilstm_runs)
attn_train_loss, attn_val_loss, attn_val_f1 = parse_runs(bilstm_attn_runs)

# Helper to calculate average metrics by epoch (up to max epoch)
def get_stats(run_data):
    max_epochs = max(len(r) for r in run_data)
    epochs = np.arange(1, max_epochs + 1)
    means = []
    stds = []
    
    for epoch_idx in range(max_epochs):
        vals = [r[epoch_idx] for r in run_data if epoch_idx < len(r)]
        means.append(np.mean(vals))
        stds.append(np.std(vals))
        
    return epochs, np.array(means), np.array(stds)

b_epochs, b_train_mean, b_train_std = get_stats(bilstm_train_loss)
_, b_val_mean, b_val_std = get_stats(bilstm_val_loss)
_, b_f1_mean, b_f1_std = get_stats(bilstm_val_f1)

a_epochs, a_train_mean, a_train_std = get_stats(attn_train_loss)
_, a_val_mean, a_val_std = get_stats(attn_val_loss)
_, a_f1_mean, a_f1_std = get_stats(attn_val_f1)

# Color Scheme: Elegant Purple Palette
color_bilstm_train = "#a855f7" # Purple-500
color_bilstm_val = "#7c3aed"   # Violet-600
color_attn_train = "#ec4899"   # Pink-500
color_attn_val = "#db2777"     # Pink-600

# ----------------- PLOT 1: LOSS CURVES COMPARISON -----------------
plt.figure(figsize=(10, 6))

# BiLSTM
plt.plot(b_epochs, b_train_mean, label="BiLSTM - Train Loss", color=color_bilstm_train, linestyle="--", marker="o")
plt.plot(b_epochs, b_val_mean, label="BiLSTM - Val Loss", color=color_bilstm_val, linestyle="-", marker="s")
plt.fill_between(b_epochs, b_val_mean - b_val_std, b_val_mean + b_val_std, color=color_bilstm_val, alpha=0.15)

# BiLSTM + Attention
plt.plot(a_epochs, a_train_mean, label="BiLSTM+Attn - Train Loss", color=color_attn_train, linestyle="--", marker="o")
plt.plot(a_epochs, a_val_mean, label="BiLSTM+Attn - Val Loss", color=color_attn_val, linestyle="-", marker="s")
plt.fill_between(a_epochs, a_val_mean - a_val_std, a_val_mean + a_val_std, color=color_attn_val, alpha=0.15)

plt.title("So Sánh Quá Trình Huấn Luyện (Loss Curves) - BiLSTM vs BiLSTM+Attention", pad=15)
plt.xlabel("Epoch")
plt.ylabel("Loss (Cross-Entropy)")
plt.xticks(np.arange(1, 11))
plt.legend(frameon=True, facecolor="white", edgecolor="none")
plt.tight_layout()
loss_plot_path = os.path.join(plots_dir, "training_loss_comparison.png")
plt.savefig(loss_plot_path, dpi=300)
plt.close()
print(f"Saved: {loss_plot_path}")

# ----------------- PLOT 2: VALIDATION F1 COMPARISON -----------------
plt.figure(figsize=(10, 6))

plt.plot(b_epochs, b_f1_mean, label="BiLSTM (Baseline)", color=color_bilstm_val, linestyle="-", marker="o", linewidth=2)
plt.fill_between(b_epochs, b_f1_mean - b_f1_std, b_f1_mean + b_f1_std, color=color_bilstm_val, alpha=0.15)

plt.plot(a_epochs, a_f1_mean, label="BiLSTM + Attention", color=color_attn_val, linestyle="-", marker="s", linewidth=2)
plt.fill_between(a_epochs, a_f1_mean - a_f1_std, a_f1_mean + a_f1_std, color=color_attn_val, alpha=0.15)

plt.title("Biến Thiên Chỉ Số F1-Macro Trên Tập Validation", pad=15)
plt.xlabel("Epoch")
plt.ylabel("F1-Macro Score")
plt.xticks(np.arange(1, 11))
plt.ylim(0.87, 0.91)
plt.legend(frameon=True, facecolor="white", edgecolor="none", loc="lower right")
plt.tight_layout()
f1_plot_path = os.path.join(plots_dir, "val_f1_comparison.png")
plt.savefig(f1_plot_path, dpi=300)
plt.close()
print(f"Saved: {f1_plot_path}")

# ----------------- PLOT 3: TEST METRICS COMPARISON (BAR CHART) -----------------
# Read test metrics summary
with open(os.path.join(exp_dir, "bilstm_multi_run_summary.json"), "r", encoding="utf-8") as f:
    b_summary = json.load(f)
with open(os.path.join(exp_dir, "bilstm_attention_multi_run_summary.json"), "r", encoding="utf-8") as f:
    a_summary = json.load(f)

metrics_names = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
vietnamese_names = ["Accuracy", "F1 Macro", "Precision", "Recall"]

b_means = [b_summary["metrics"][m]["avg"] for m in metrics_names]
b_stds = [b_summary["metrics"][m]["std"] for m in metrics_names]

a_means = [a_summary["metrics"][m]["avg"] for m in metrics_names]
a_stds = [a_summary["metrics"][m]["std"] for m in metrics_names]

x = np.arange(len(vietnamese_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, b_means, width, yerr=b_stds, label='BiLSTM (Baseline)', color="#7c3aed", capsize=5, alpha=0.85)
rects2 = ax.bar(x + width/2, a_means, width, yerr=a_stds, label='BiLSTM + Attention', color="#ec4899", capsize=5, alpha=0.85)

ax.set_title('So Sánh Hiệu Năng Trên Tập Kiểm Tra (Test Set)', pad=15)
ax.set_ylabel('Score')
ax.set_xticks(x)
ax.set_xticklabels(vietnamese_names)
ax.set_ylim(0.85, 0.92)
ax.legend(frameon=True, facecolor="white", edgecolor="none", loc="lower right")

# Add labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
test_plot_path = os.path.join(plots_dir, "test_metrics_comparison.png")
plt.savefig(test_plot_path, dpi=300)
plt.close()
print(f"Saved: {test_plot_path}")
