import os
import sys
import platform
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Ensure project root is in path
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup directories
OUTPUT_DIR = PROJECT_ROOT / "reports" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_font_path():
    """Find a standard Vietnamese-supporting font path depending on OS."""
    system = platform.system()
    if system == "Windows":
        paths = [
            "C:\\Windows\\Fonts\\arial.ttf",
            "C:\\Windows\\Fonts\\tahoma.ttf",
            "C:\\Windows\\Fonts\\times.ttf"
        ]
        for p in paths:
            if os.path.exists(p):
                return p
    elif system == "Linux":
        paths = [
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ]
        for p in paths:
            if os.path.exists(p):
                return p
    return None

def main():
    dataset_path = PROJECT_ROOT / "data" / "raw" / "ntc_scv.csv"
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}. Please run download script first.")
        sys.exit(1)
        
    print("Loading dataset...")
    df = pd.read_csv(dataset_path)
    
    # ----------------------------------------------------
    # 1. Biểu đồ tròn biểu diễn tần suất phân bổ nhãn cảm xúc
    # ----------------------------------------------------
    print("1. Generating sentiment distribution pie chart...")
    label_counts = df['label'].value_counts()
    
    # Map label keys to text
    labels_map = {0: "Tiêu cực", 1: "Tích cực"}
    labels = [labels_map[idx] for idx in label_counts.index]
    colors = ["#48bb78" if idx == 1 else "#f6e05e" if idx == -1 else "#f56565" for idx in label_counts.index]
    
    plt.figure(figsize=(6, 6))
    plt.pie(
        label_counts, 
        labels=labels, 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=colors,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2, 'antialiased': True},
        textprops={'fontsize': 12, 'weight': 'bold'}
    )
    plt.title("Phân bổ nhãn cảm xúc trong bộ dữ liệu NTC-SCV", fontsize=14, weight='bold', pad=20)
    plt.tight_layout()
    pie_chart_path = OUTPUT_DIR / "sentiment_pie_chart.png"
    plt.savefig(pie_chart_path, dpi=200)
    plt.close()
    print(f"   Saved pie chart -> {pie_chart_path}")
    
    # ----------------------------------------------------
    # 2. Biểu đồ mật độ kết hợp lược đồ phân bố histogram độ dài câu
    # ----------------------------------------------------
    print("2. Generating sentence length histogram and density plot...")
    # Calculate word count for each text (split by space)
    df['word_count'] = df['text'].astype(str).apply(lambda x: len(x.split()))
    
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df, 
        x="word_count", 
        kde=True, 
        bins=60, 
        color="#3182ce", 
        line_kws={"linewidth": 2.5}
    )
    plt.title("Phân bố độ dài bình luận (số từ) trong tập dữ liệu NTC-SCV", fontsize=14, weight='bold', pad=15)
    plt.xlabel("Số lượng từ trong một câu bình luận", fontsize=12)
    plt.ylabel("Tần suất xuất hiện (Số câu)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Adjust axis limit to exclude outliers and zoom into density
    plt.xlim(0, df['word_count'].quantile(0.99)) # Cut off top 1% outliers for visual clarity
    plt.tight_layout()
    dist_chart_path = OUTPUT_DIR / "sentence_length_distribution.png"
    plt.savefig(dist_chart_path, dpi=200)
    plt.close()
    print(f"   Saved distribution chart -> {dist_chart_path}")
    
    # ----------------------------------------------------
    # 2b. Biểu đồ so sánh phân bố độ dài câu tích cực và tiêu cực
    # ----------------------------------------------------
    print("2b. Generating sentence length comparison chart (Positive vs. Negative)...")
    # Map label to text for visualization
    df['Sắc thái'] = df['label'].map({1: 'Tích cực', 0: 'Tiêu cực'})
    
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df, 
        x="word_count", 
        hue="Sắc thái", 
        kde=True, 
        bins=60, 
        palette={"Tích cực": "#48bb78", "Tiêu cực": "#f56565"},
        alpha=0.4,
        element="step",
        stat="density",
        common_norm=False,
        line_kws={"linewidth": 2}
    )
    plt.title("So sánh phân bố độ dài bình luận Tích cực vs. Tiêu cực", fontsize=14, weight='bold', pad=15)
    plt.xlabel("Số lượng từ trong một câu bình luận", fontsize=12)
    plt.ylabel("Mật độ phân bố (Density)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Adjust axis limit to exclude outliers
    plt.xlim(0, df['word_count'].quantile(0.99))
    plt.tight_layout()
    comparison_chart_path = OUTPUT_DIR / "sentence_length_comparison.png"
    plt.savefig(comparison_chart_path, dpi=200)
    plt.close()
    print(f"   Saved comparison chart -> {comparison_chart_path}")
    
    # ----------------------------------------------------
    # 3. Hệ thống đám mây từ ngữ (Word Cloud) cho tích cực và tiêu cực
    # ----------------------------------------------------
    print("3. Generating positive and negative Word Clouds...")
    font_path = get_font_path()
    if font_path:
        print(f"   Using font for WordCloud: {font_path}")
    else:
        print("   Warning: Could not find system font, fallback to default (Vietnamese diacritics might render incorrectly).")
        
    # Standard Vietnamese stopwords
    vietnamese_stopwords = {
        "và", "của", "được", "có", "là", "trong", "cho", "để", "với", "các", "này", "ra", "ở", "nào", "đó", 
        "những", "nếu", "thì", "mà", "như", "nhưng", "vì", "nên", "lại", "đã", "đang", "sẽ", "còn", "cũng", 
        "mình", "quán", "món", "ăn", "uống", "nói", "làm", "nhận", "khi", "sau", "trước", "về", "lên", "xuống",
        "hơn", "chỉ", "vào", "qua", "lại", "nhất", "đến", "nơi", "đi", "đây", "bên", "thấy", "lúc", "vẫn"
    }
    
    # Split positive (1) and negative (0) texts
    pos_reviews = df[df['label'] == 1]['text'].astype(str).tolist()
    neg_reviews = df[df['label'] == 0]['text'].astype(str).tolist()
    
    # Limit number of reviews to concatenate for performance
    pos_text = " ".join(pos_reviews[:15000])
    neg_text = " ".join(neg_reviews[:15000])
    
    # Positive Word Cloud
    pos_wc = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        max_words=150, 
        stopwords=vietnamese_stopwords, 
        font_path=font_path,
        colormap='YlGnBu'
    ).generate(pos_text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(pos_wc, interpolation='bilinear')
    plt.axis('off')
    plt.title("Đám mây từ ngữ Tích cực (Positive Word Cloud)", fontsize=14, weight='bold', pad=15)
    plt.tight_layout(pad=0)
    pos_wc_path = OUTPUT_DIR / "positive_wordcloud.png"
    plt.savefig(pos_wc_path, dpi=200)
    plt.close()
    print(f"   Saved positive wordcloud -> {pos_wc_path}")
    
    # Negative Word Cloud
    neg_wc = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        max_words=150, 
        stopwords=vietnamese_stopwords, 
        font_path=font_path,
        colormap='OrRd'
    ).generate(neg_text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(neg_wc, interpolation='bilinear')
    plt.axis('off')
    plt.title("Đám mây từ ngữ Tiêu cực (Negative Word Cloud)", fontsize=14, weight='bold', pad=15)
    plt.tight_layout(pad=0)
    neg_wc_path = OUTPUT_DIR / "negative_wordcloud.png"
    plt.savefig(neg_wc_path, dpi=200)
    plt.close()
    print(f"   Saved negative wordcloud -> {neg_wc_path}")
    
    print("\n✅ Dataset visualization complete! Images saved in: reports/plots/")

if __name__ == "__main__":
    main()
