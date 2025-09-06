import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from matplotlib.lines import Line2D

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Microsoft YaHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

input_folder = "7_疾病暴露資料"
output_folder = "9-5_importance_lower"
csv_output_folder = "9-5_importance_lower/csv"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(csv_output_folder, exist_ok=True)

target_diseases = ["急性Bronchitis", "慢性Bronchitis", "Pneumonia", "氣喘"]
pollutants = ["O3", "PM10", "PM25", "SO2"]

for filename in os.listdir(input_folder):
    if not filename.endswith(".csv"):
        continue

    disease_name = filename.replace(".csv", "")
    if disease_name not in target_diseases:
        continue

    # 讀檔與基本清理
    df = pd.read_csv(os.path.join(input_folder, filename))
    df = df[df["year"].between(2016, 2019)].copy()
    df = df[["region", "year", "week", "case_per_capita(‰)"] + pollutants].dropna()
    df["key"] = df["region"] + "_" + df["year"].astype(str) + "_" + df["week"].astype(str)

    for lag in range(5):  # lag: 0~4
        cumulative_df = None

        # 針對 0..lag 的每個 offset，建立位移後的暴露表，並逐步 inner merge
        for offset in range(lag + 1):
            temp = df.copy()
            temp["week_shifted"] = temp["week"] + offset
            temp["year_shifted"] = temp["year"] + (temp["week_shifted"] - 1) // 52
            temp["week_shifted"] = (temp["week_shifted"] - 1) % 52 + 1
            temp["key_shifted"] = (
                temp["region"] + "_" +
                temp["year_shifted"].astype(str) + "_" +
                temp["week_shifted"].astype(str)
            )

            # 只保留 key 與該 offset 的污染物欄位，並加上 offset 後綴
            temp = temp[["key_shifted"] + pollutants].copy()
            temp = temp.rename(columns={p: f"{p}_{offset}" for p in pollutants})

            if cumulative_df is None:
                cumulative_df = temp
            else:
                # ✅ inner merge：只有所有 offset 都存在的 key 才會被保留
                cumulative_df = cumulative_df.merge(temp, on="key_shifted", how="inner")

        # 若這個 lag 沒有任何完整窗口，跳過
        if cumulative_df is None or cumulative_df.empty:
            continue

        # 計算各污染物的累積和（因為是 inner merge，這裡不會有 NaN）
        for p in pollutants:
            cols = [f"{p}_{o}" for o in range(lag + 1)]
            # 保險起見：若缺欄位（理論上不會）就跳過
            cols = [c for c in cols if c in cumulative_df.columns]
            if len(cols) == 0:
                continue
            cumulative_df[p + "_sum"] = cumulative_df[cols].sum(axis=1)

        # 與目標值對齊（使用當週的就醫率）
        df_target = df.copy()
        df_target["key_shifted"] = df_target["key"]
        merged = pd.merge(
            cumulative_df[["key_shifted"] + [p + "_sum" for p in pollutants]],
            df_target[["key_shifted", "case_per_capita(‰)"]],
            on="key_shifted",
            how="inner"
        )

        if merged.empty:
            continue

        # 解析 key → region/year/week
        merged[["region", "year", "week"]] = merged["key_shifted"].str.split("_", expand=True)
        merged["year"] = merged["year"].astype(int)
        merged["week"] = merged["week"].astype(int)

        # 依照地區、年份、週數排序
        merged = merged.sort_values(by=["region", "year", "week"]).reset_index(drop=True)

        if len(merged) < 20:
            continue

        # ==== 輸出 CSV（排序 + 指定欄位順序）====
        csv_path = os.path.join(csv_output_folder, f"{disease_name}_lag0to{lag}_cumulative_data.csv")
        output_cols = ["region", "year", "week", "case_per_capita(‰)"] + [p + "_sum" for p in pollutants]
        merged[output_cols].to_csv(csv_path, index=False, encoding='utf_8_sig')

        # ==== 建模與繪圖 ====
        X = merged[[p + "_sum" for p in pollutants]].copy()
        X.columns = pollutants
        y = merged["case_per_capita(‰)"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        result = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42, scoring="r2")
        sorted_idx = result.importances_mean.argsort()

        plt.figure(figsize=(8, 6))
        plt.barh(
            range(len(sorted_idx)),
            result.importances_mean[sorted_idx],
            xerr=result.importances_std[sorted_idx],
            align="center",
            color="#3399CC"
        )
        plt.yticks(range(len(sorted_idx)), [X.columns[i] for i in sorted_idx])
        plt.xlabel("Permutation Importance (R² decrease)")
        plt.title(f"{disease_name} 特徵重要性（lag0~{lag} 累積）")

        error_bar_legend = Line2D(
            [0], [0],
            color='black',
            lw=2,
            label='黑線代表標準差\n反映特徵重要性穩定性'
        )
        plt.legend(handles=[error_bar_legend], loc='lower right', frameon=True)

        plt.tight_layout()
        output_path = os.path.join(output_folder, f"{disease_name}_cumulative_lag0to{lag}_importance.png")
        plt.savefig(output_path, dpi=300)
        plt.close()

print("✅ 完成：各疾病 lag0~4 的特徵重要性圖與累積資料（已剔除不完整累加 + 各 region 頭尾各 lag 週）")
