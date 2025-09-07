import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import spearmanr, pearsonr
import re

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Microsoft YaHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# 區域顏色對應
region_color_map = {
    "高屏": "#AA04AA",        # 紫
    "中彰投": '#FF0000',      # 紅
    "雲嘉南": '#FFA500',      # 橘
    "北北基桃竹苗": '#FFFF00', # 黃
    "宜花東": "#23B623"       # 綠
}

# 空氣因子列表（使用 *_sum 欄位）
pollutants = ["O3_sum", "PM10_sum", "PM25_sum", "SO2_sum"]

# 需要處理的資料夾設定
datasets = [
    {"input": "1-1_importance_upper/csv", "output": "2-1_scatter_upper"},
    {"input": "1-2_importance_lower/csv", "output": "2-2_scatter_lower"}
]

for data in datasets:
    input_folder = data["input"]
    base_output_folder = data["output"]

    print(f"\n🔹 處理資料夾：{input_folder} → {base_output_folder}")

    # 收集 Spearman 結果
    results = {}

    for filename in os.listdir(input_folder):
        if not filename.endswith(".csv"):
            continue

        match = re.match(r"(.+)_lag0to(\d+)_cumulative_data\.csv", filename)
        if not match:
            continue

        disease_name, lag_str = match.groups()
        lag = int(lag_str)

        df = pd.read_csv(os.path.join(input_folder, filename)).dropna()

        for pollutant in pollutants:
            if pollutant not in df.columns:
                continue

            merged = df[["region", "year", "week", "case_per_capita(‰)", pollutant]].copy()
            if len(merged) < 10:
                continue

            pearson = pearsonr(merged[pollutant], merged["case_per_capita(‰)"])[0]
            spearman = spearmanr(merged[pollutant], merged["case_per_capita(‰)"])[0]

            results.setdefault(disease_name, {}).setdefault(pollutant.replace("_sum",""), []).append({
                "lag": lag,
                "spearman": spearman,
                "pearson": pearson,
                "data": merged
            })

    # 🚀 根據 Spearman 排序後繪圖
    for disease, pollutants_dict in results.items():
        for pollutant, items in pollutants_dict.items():
            # 保留 Spearman 絕對值最高前 5
            sorted_items = sorted(items, key=lambda x: abs(x["spearman"]), reverse=True)[:5]

            output_plot_folder = os.path.join(base_output_folder, pollutant)
            os.makedirs(output_plot_folder, exist_ok=True)

            for rank, r in enumerate(sorted_items, 1):
                merged = r["data"]
                lag = r["lag"]

                X = merged[pollutant + "_sum"].values.reshape(-1, 1) if pollutant+"_sum" in merged else merged[pollutant].values.reshape(-1,1)
                y = merged["case_per_capita(‰)"].values
                colors = merged["region"].map(region_color_map)

                plt.figure(figsize=(7, 6))
                plt.scatter(X, y, alpha=0.6, c=colors)

                if pollutant == "O3" or pollutant == "SO2":
                    plt.xlabel(f"{pollutant} 累積({lag+1}週 ppm)")
                elif pollutant == "PM10" or pollutant == "PM25":
                    plt.xlabel(f"{pollutant} 累積({lag+1}週 μg/m³)")

                plt.ylabel("就診人數(‰)")
                plt.title(f"{disease}：{pollutant} lag累積={lag}週 散布圖")

                stats_text = (
                    f"          總體\n"
                    f"Pearson: {r['pearson']:.3f}\n"
                    f"Spearman: {r['spearman']:.3f}"
                )
                plt.text(
                    0.02, 0.98,
                    stats_text,
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    horizontalalignment='left',
                    fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
                )

                legend_elements = []
                for region, color in region_color_map.items():
                    sub = merged[merged["region"] == region]
                    if len(sub) < 2:
                        continue
                    X_sub = sub[[pollutant + "_sum"]].values
                    y_sub = sub["case_per_capita(‰)"].values
                    reg = LinearRegression().fit(X_sub, y_sub)
                    slope = reg.coef_[0]
                    x_range = np.linspace(X_sub.min(), X_sub.max(), 100).reshape(-1, 1)
                    y_pred = reg.predict(x_range)
                    plt.plot(x_range, y_pred, color=color, linewidth=1.8)
                    legend_elements.append(
                        Line2D([0], [0], marker='o', color='w', label=f"{region} (m={slope:.4f})",
                               markerfacecolor=color, markersize=8)
                    )

                plt.legend(handles=legend_elements, title="區域", loc='lower right', frameon=True)
                plt.tight_layout()
                plot_filename = f"{disease}_{pollutant}_rank{rank}_lag{lag}_accum.png"
                save_path = os.path.join(output_plot_folder, plot_filename)
                plt.savefig(save_path, dpi=300)
                plt.close()

                print(f"✅ 輸出 {plot_filename} (Spearman={r['spearman']:.3f})")

    print(f"\n🎯 完成資料夾 {input_folder} 的散布圖輸出")
