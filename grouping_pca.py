import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 載入資料
df = pd.read_excel("summary.xlsx")

# 定義六群
groups = {
    "trend": ["MA_5","MA_10","MA_20","MA_60","EMA_12","EMA_26","MACD","MACD_signal","MACD_hist"],
    "momentum": ["RSI_14","K_9","D_3","Williams_%R_14","ROC_10"],
    "volatility": ["BB_upper","BB_lower","BB_width","ATR_14"],
    "volume": ["Vol_MA_10","Vol_MA_20","OBV","AD","VRSI_14"],
    "macro": ["market_value_change","market_volume_change","foreign_momentum","institutional_momentum","market_heat"],
    "chip": ["外資持股比例日變化(pp)","外資持股比例5日變化(pp)","外資持股比例20日變化(pp)",
             "投信持股比例日變化(pp)","投信持股比例5日變化(pp)","投信持股比例20日變化(pp)",
             "外資買賣超市值(百萬)_zscore","投信買賣超市值(百萬)_zscore","自營買賣超市值(百萬)_zscore",
             "chip_concentration"]
}

writer = pd.ExcelWriter("pca_group_results.xlsx", engine="openpyxl")

# 存放解釋變異比例
explained_summary = {}

for g, cols in groups.items():
    cols = [c for c in cols if c in df.columns]
    X = df[cols].fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA()
    comps = pca.fit_transform(X_scaled)
    
    # 解釋變異比例
    explained = pca.explained_variance_ratio_
    explained_summary[g] = explained
    
    # 主成分負荷量
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(len(explained))],
        index=cols
    )
    
    loadings.to_excel(writer, sheet_name=f"{g}_loadings")

rows = []
for g, vals in explained_summary.items():
    for i, v in enumerate(vals, start=1):
        rows.append({"Group": g, "PC": f"PC{i}", "Explained_Var": v})
explained_df = pd.DataFrame(rows)
explained_df.to_excel(writer, sheet_name="explained_variance", index=False)

writer.close()


