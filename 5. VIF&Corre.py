import os
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns

# 空氣因子列表
pollutants = ["NO", "NO2", "NOx", "O3", "PM25", "PM10", "SO2"]

# 需要處理的資料夾
input_folder = "0_疾病暴露資料"
output_folder = "5_VIF_and_Corr_results"
os.makedirs(output_folder, exist_ok=True)

VIF_THRESHOLD = 5  # 共線性警告閾值

print(f"🔹 讀取資料夾：{input_folder}")

all_data = []

# ===== 讀取所有 CSV 檔案 =====
for filename in os.listdir(input_folder):
    if not filename.endswith(".csv"):
        continue

    disease_name = filename.replace(".csv", "")
    df = pd.read_csv(os.path.join(input_folder, filename))

    # 保留必要欄位
    required_cols = ["region", "year", "week"] + pollutants
    df = df[required_cols].dropna(subset=pollutants)
    df["disease"] = disease_name
    all_data.append(df)

if not all_data:
    print("⚠️ 沒有找到有效 CSV 檔案")
else:
    merged_df = pd.concat(all_data, ignore_index=True)

    # ===== 計算 VIF =====
    vif_data = merged_df[pollutants].copy()
    vif_data = vif_data.dropna()  # 移除缺失值

    vif_df = pd.DataFrame()
    vif_df["variable"] = pollutants
    vif_df["VIF"] = [variance_inflation_factor(vif_data.values, i) for i in range(len(pollutants))]
    vif_df["high_collinearity"] = vif_df["VIF"] > VIF_THRESHOLD

    # 儲存 VIF 結果
    output_vif_file = os.path.join(output_folder, "VIF_all_diseases.csv")
    vif_df.to_csv(output_vif_file, index=False)
    print(f"✅ VIF 計算完成，結果儲存至：{output_vif_file}\n")
    print(vif_df)

    # ===== 畫相關係數矩陣熱力圖 =====
    corr_df = merged_df[pollutants].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_df,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"label": "Correlation"}
    )
    plt.title("Pollutant Correlation Matrix")
    plt.tight_layout()

    output_corr_img = os.path.join(output_folder, "Correlation_matrix.png")
    plt.savefig(output_corr_img, dpi=300)
    plt.close()
    print(f"✅ Correlation Matrix 圖片已儲存至：{output_corr_img}")
