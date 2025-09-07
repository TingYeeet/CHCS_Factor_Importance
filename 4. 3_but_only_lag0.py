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
    {"input": "1-1_importance_upper/csv", "output": "4-1_region_upper_only_0"},
    {"input": "1-2_importance_lower/csv", "output": "4-2_region_lower_only_0"}
]

for data in datasets:
    input_folder = data["input"]
    base_output_folder = data["output"]

    print(f"\n🔹 處理資料夾：{input_folder} → {base_output_folder}")
    # ===== 第一步：收集所有 Spearman 結果 =====
    results = []  # [(disease, pollutant, region, lag, spearman, pearson, data)]

    for filename in os.listdir(input_folder):
        if not filename.endswith(".csv"):
            continue

        match = re.match(r"(.+)_lag0to(\d+)_cumulative_data\.csv", filename)
        if not match:
            continue

        disease_name, lag_str = match.groups()
        lag = int(lag_str)

        if lag != 0:
            continue

        df = pd.read_csv(os.path.join(input_folder, filename)).dropna()

        for pollutant in pollutants:
            if pollutant not in df.columns:
                continue

            for region in region_color_map.keys():
                sub = df[df["region"] == region]
                if len(sub) < 5:
                    continue
                spearman = spearmanr(sub[pollutant], sub["case_per_capita(‰)"])[0]
                pearson = pearsonr(sub[pollutant], sub["case_per_capita(‰)"])[0]
                results.append({
                    "disease": disease_name,
                    "pollutant": pollutant,
                    "region": region,
                    "lag": lag,
                    "spearman": spearman,
                    "pearson": pearson,
                    "data": sub
                })

    # ===== 第二步：保留每地區每污染物 Spearman 絕對值最高的 1 =====
    res_df = pd.DataFrame(results)
    res_df["abs_spearman"] = res_df["spearman"].abs()

    top_df = (
        res_df
        .sort_values(["disease", "pollutant", "region", "abs_spearman"], ascending=[True, True, True, False])
        .groupby(["disease", "pollutant", "region"])
        .head(1)
    )

    # ===== 第三步：繪圖並輸出 =====
    for _, row in top_df.iterrows():
        disease_name = row["disease"]
        pollutant = row["pollutant"]
        region = row["region"]
        lag = row["lag"]
        merged = row["data"]

        X = merged[[pollutant]].values
        y = merged["case_per_capita(‰)"].values

        # 迴歸線
        reg = LinearRegression().fit(X, y)
        slope = reg.coef_[0]
        x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_pred = reg.predict(x_range)

        # 圖片輸出資料夾：依空氣因子
        output_plot_folder = os.path.join(base_output_folder, pollutant.replace("_sum",""))
        os.makedirs(output_plot_folder, exist_ok=True)

        plt.figure(figsize=(7, 6))
        plt.scatter(X, y, alpha=0.6, c=region_color_map[region], label=region)
        plt.plot(x_range, y_pred, color=region_color_map[region], linewidth=2)

        if pollutant == "O3" or pollutant == "SO2":
            plt.xlabel(f"{pollutant.replace('_sum','')} 累積({lag+1}週 ppm)")
        elif pollutant == "PM10" or pollutant == "PM25":
            plt.xlabel(f"{pollutant.replace('_sum','')} 累積({lag+1}週 μg/m³)")

        plt.ylabel("就診人數(‰)")
        plt.title(f"{disease_name} - {region}\n{pollutant.replace('_sum','')} lag={lag} 散布圖")

        stats_text = (
            f"Spearman: {row['spearman']:.3f}\n"
            f"Pearson: {row['pearson']:.3f}\n"
            f"斜率 m: {slope:.4f}"
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

        plt.legend(loc="lower right")
        plt.tight_layout()

        plot_filename = f"{disease_name}_{region}_{pollutant.replace('_sum','')}_lag{lag}.png"
        plt.savefig(os.path.join(output_plot_folder, plot_filename), dpi=300)
        plt.close()

        print(f"✅ {disease_name} - {region} - {pollutant.replace('_sum','')} lag{lag} 完成")
    print(f"\n🎯 完成資料夾 {input_folder} 的單區域lag 0散布圖輸出")

print("🎉 所有 Spearman 最高的散布圖已完成")
