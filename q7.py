import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import linregress

CSV_PATH = "food_orders_new_delhi_ext2.csv"  # ggf. Pfad anpassen
df = pd.read_csv(CSV_PATH)

# ----------------------------
# 0) Aufbereitung
# ----------------------------
for col in ["Order Value", "Commission Fee", "Refunds/Chargebacks"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["Refunds/Chargebacks"] = df["Refunds/Chargebacks"].fillna(0)
df["refund_flag"] = df["Refunds/Chargebacks"] > 0

# Nur Zeilen mit relevanten Werten
df = df[df["Commission Fee"].notna() & df["Order Value"].notna()].copy()

# ----------------------------
# 1) Häufigkeit der Provisionsgrößen
# ----------------------------
fee_counts = df["Commission Fee"].value_counts().sort_index()
fee_pct = fee_counts / fee_counts.sum() * 100

# plt.figure()
# plt.bar(fee_counts.index.astype(int).astype(str), fee_counts.values)
# plt.title("Häufigkeit der Provisionsgrößen (Commission Fee)")
# plt.xlabel("Provisionsgebühr (Fixwert)")
# plt.ylabel("Anzahl Bestellungen")
# plt.xticks(rotation=90)

for i, fee in enumerate(fee_counts.index):
    v = int(fee_counts.loc[fee])
    p = fee_pct.loc[fee]
    plt.text(i, v + 2, f"{v} ({p:.1f}%)", ha="center", va="bottom", fontsize=8)

# Optional: wenn zu viele Ausprägungen -> zusammengefasste Histogramm-Variante
plt.figure()
plt.hist(df["Commission Fee"].dropna(), bins=20)
plt.title("Histogramm der Provisionsgebühr")
plt.xlabel("Provisionsgebühr")
plt.ylabel("Anzahl Bestellungen")
plt.tight_layout()
plt.show()

# ----------------------------
# 2) Provision vs durchschnittliche Bestellgröße
# ----------------------------
# 2a) Scatter + lineare Regression (Order-Level)
x = df["Commission Fee"]
y = df["Order Value"]
m = x.notna() & y.notna()
lr = linregress(x[m], y[m])

plt.figure()
plt.scatter(x[m], y[m], s=12)
plt.title("Bestellwert vs. Provisionsgebühr ")
plt.xlabel("Provisionsgebühr")
plt.ylabel("Bestellwert")

x_line = np.linspace(x[m].min(), x[m].max(), 100)
y_line = lr.intercept + lr.slope * x_line
plt.plot(x_line, y_line)
plt.tight_layout()
plt.show()

print("\n=== Provision (Fix) -> Bestellwert ===")
print(f"Regression: slope={lr.slope:.6f}, intercept={lr.intercept:.2f}, R²={lr.rvalue**2:.4f}")
r_p, p_p = stats.pearsonr(x[m], y[m])
r_s, p_s = stats.spearmanr(x[m], y[m])
print(f"Pearson r={r_p:.4f} (p={p_p:.4g}), Spearman ρ={r_s:.4f} (p={p_s:.4g})")


df["commission_fee_bucket"] = pd.qcut(df["Commission Fee"], 10, labels=["D1", "D2", "D3", "D4 ", "D5", "D6", "D7", "D8", "D9", "D10"])
bucket_order = ["D1", "D2", "D3", "D4 ", "D5", "D6", "D7", "D8", "D9", "D10"]

bucket_summary = (
    df.groupby("commission_fee_bucket", observed=True)
      .agg(
          orders=("Order ID", "count"),
          median_commission_fee=("Commission Fee", "median"),
          avg_order_value=("Order Value", "mean"),
      )
      .reindex(bucket_order)
      .reset_index()
)

bucket_summary["median_commission_fee"] = bucket_summary["median_commission_fee"].round(1)
bucket_summary["avg_order_value"] = bucket_summary["avg_order_value"].round(2)

print("\n=== Ø Bestellwert nach Provisions-Bucket (Quartile) ===")
print(bucket_summary.to_string(index=False))

plt.figure()
plt.bar(bucket_summary["commission_fee_bucket"].astype(str), bucket_summary["avg_order_value"].values)
plt.title("Avg Bestellwert nach Provisionsdezil")
plt.xlabel("Provisionsdezile")
plt.ylabel("Avg Bestellwert")
for i, v in enumerate(bucket_summary["avg_order_value"].values):
    plt.text(i, v + 5, f"{v:.0f}", ha="center", va="bottom")
plt.tight_layout()
plt.show()

# ----------------------------
# 3) Provision vs Häufigkeit der Rückerstattungen
# ----------------------------
bucket_refund = (
    df.groupby("commission_fee_bucket", observed=True)["refund_flag"]
      .agg(orders="count", refund_rate="mean")
      .reindex(bucket_order)
      .reset_index()
)
bucket_refund["refund_rate_%"] = (bucket_refund["refund_rate"] * 100).round(1)

print("\n=== Refund-Rate nach Provisions-Bucket (Quartile) ===")
print(bucket_refund[["commission_fee_bucket", "orders", "refund_rate_%"]].to_string(index=False))

plt.figure()
plt.bar(bucket_refund["commission_fee_bucket"].astype(str), bucket_refund["refund_rate_%"].values)
plt.title("Refund-Rate nach Provisionsdezil")
plt.xlabel("Provisionsdezile")
plt.ylabel("Refund-Rate (%)")
for i, v in enumerate(bucket_refund["refund_rate_%"].values):
    plt.text(i, v + 0.5, f"{v:.1f}%", ha="center", va="bottom")
plt.tight_layout()
plt.show()

# Zusätzlich: Test Refund ja/nein ~ Commission Fee (Punkt-biserial / Pearson mit 0/1)
rf = df["refund_flag"].astype(int)
m2 = df["Commission Fee"].notna() & rf.notna()
r_rf, p_rf = stats.pearsonr(df.loc[m2, "Commission Fee"], rf.loc[m2])
print("\n=== Provision (Fix) -> Refund (ja/nein) ===")
print(f"Point-biserial/Pearson r={r_rf:.4f}, p={p_rf:.4g}")

