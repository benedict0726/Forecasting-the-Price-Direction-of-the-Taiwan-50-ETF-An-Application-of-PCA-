import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

n_pc_keep = {"trend":2,"momentum":2,"volatility":1,"volume":3,"macro":3,"chip":6}

#  1 載入資料 
df = pd.read_excel("summary.xlsx")
date_col = None
for c in df.columns:
    cl = str(c).lower()
    if cl in {"date","年月日","日期","交易日"} or "date" in cl:
        date_col = c
        break
if date_col is not None:
    df["Date"] = pd.to_datetime(df[date_col])
else:
    df["Date"] = pd.NaT

#2 定義六群 
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

#  3 輸出
writer = pd.ExcelWriter("pca_group_results.xlsx", engine="openpyxl")
explained_summary = {}

X_pcs = pd.DataFrame(index=df.index)

for g, cols in groups.items():
    cols = [c for c in cols if c in df.columns]
    if len(cols) == 0:
        continue

    X = df[cols].fillna(0)

    # 標準化 + PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()                     # 全部主成分
    comps = pca.fit_transform(X_scaled)   

    # 原本就有的輸出 
    explained = pca.explained_variance_ratio_
    explained_summary[g] = explained

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(len(explained))],
        index=cols
    )
    loadings.to_excel(writer, sheet_name=f"{g}_loadings")

    # 把要保留的前 K 個 PC 分數存進 X_pcs 
    k = n_pc_keep.get(g, 1)                    
    k = min(k, comps.shape[1])                
    for i in range(k):
        X_pcs[f"{g}_PC{i+1}"] = comps[:, i]

#  4 explained_variance 
rows = []
for g, vals in explained_summary.items():
    cum = 0.0
    for i, v in enumerate(vals, start=1):
        cum += float(v)
        rows.append({"Group": g, "PC": f"PC{i}", "Explained_Var": v, "Cumulative": cum})
explained_df = pd.DataFrame(rows)
explained_df.to_excel(writer, sheet_name="explained_variance", index=False)

writer.close()


#  5 輸出最終要丟給模型的 PC 矩陣 
final_out = pd.concat([df[["Date"]] if "Date" in df.columns else df.iloc[:, :0], X_pcs], axis=1)

final_out.to_excel("final_PC_dataset.xlsx", index=False)
