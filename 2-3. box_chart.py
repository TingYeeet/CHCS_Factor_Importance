import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import re
import numpy as np

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Microsoft YaHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# 顏色對應（四個疾病）
disease_colors = {
    "URI": "#1f77b4",
    "急性Rhinosinusitis": "#ff7f0e",
    "Allergic rhinitis": "#2ca02c",
    "Influenza": "#d62728"
}

# 區域清單
regions = ["高屏", "中彰投", "雲嘉南", "北北基桃竹苗", "宜花東"]

# 空氣因子列表（使用 *_sum 欄位）
pollutants = ["O3_sum", "PM10_sum", "PM25_sum", "SO2_sum"]

# 需要處理的資料夾設定
datasets = [
    {"input": "1-1_importance_upper/csv", "output": "2-3_spearmanbar_upper"},
    {"input": "1-2_importance_lower/csv", "output": "2-4_spearmanbar_lower"}
]

for data in datasets:
    input_folder = data["input"]
    base_output_folder = data["output"]
    os.makedirs(base_output_folder, exist_ok=True)

    print(f"\n🔹 處理資料夾：{input_folder} → {base_output_folder}")

    # 收集 Spearman 結果： pollutant → region → disease → [(lag, spearman)]
    results = {p.replace("_sum", ""): {r: {} for r in regions} for p in pollutants}

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

            for region in regions:
                sub = df[df["region"] == region]
                if len(sub) < 5:
                    continue
                spearman = spearmanr(sub[pollutant], sub["case_per_capita(‰)"])[0]
                results[pollutant.replace("_sum", "")][region].setdefault(disease_name, []).append((lag, spearman))

    # 🚀 繪製 Spearman 長條圖（每個污染物 * 每個地區一張圖）
    for pollutant, region_dict in results.items():
        for region, disease_dict in region_dict.items():
            plt.figure(figsize=(10, 6))

            width = 0.2
            x = np.arange(5)  # lag 0~4

            for i, (disease, lag_values) in enumerate(disease_dict.items()):
                lag_values_sorted = sorted(lag_values, key=lambda x: x[0])
                spearman_vals = [s for _, s in lag_values_sorted]

                # 確保 lag=0~4 都有值（缺的補 NaN）
                full_spearman = []
                for lag in range(5):
                    match_vals = [s for l, s in lag_values_sorted if l == lag]
                    full_spearman.append(match_vals[0] if match_vals else np.nan)

                # 畫長條
                bars = plt.bar(x + i*width, full_spearman, width=width,
                               label=disease, color=disease_colors.get(disease, None), alpha=0.7)

                # 在每個柱子上標 Spearman 值
                for bar, val in zip(bars, full_spearman):
                    if not np.isnan(val):
                        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                                 f"{val:.3f}", ha='center', va='bottom', fontsize=8)

                # 疾病折線
                plt.plot(x + i*width, full_spearman, marker="o",
                         color=disease_colors.get(disease, None))

            plt.axhline(0, color="black", linewidth=1)
            plt.xticks(x + width*len(disease_dict)/2, [f"lag {i}" for i in range(5)])
            plt.ylabel("Spearman 相關係數")
            plt.title(f"{pollutant} - {region} 各疾病 Spearman 值（lag=0~4）")
            plt.legend()

            plt.tight_layout()
            output_path = os.path.join(base_output_folder, f"{pollutant}_{region}_spearman_bar.png")
            plt.savefig(output_path, dpi=300)
            plt.close()

            print(f"✅ 輸出 {output_path}")

print("\n🎯 每個污染物在各地區的 Spearman 長條圖完成！")
