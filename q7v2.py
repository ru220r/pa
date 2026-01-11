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

# Nur valide Bestellungen (für Quote braucht man Order Value > 0)
df = df[df["Order Value"].notna() & (df["Order Value"] > 0) & df["Commission Fee"].notna()].copy()

# Provisionsquote in %
df["commission_rate_pct"] = df["Commission Fee"] / df["Order Value"] * 100

# ----------------------------
# 1) Häufigkeit / Verteilung der Provisionsquote
# ----------------------------
plt.figure()
plt.hist(df["commission_rate_pct"].dropna(), bins=25)
plt.title("Verteilung der Provisionsquote")
plt.xlabel("Provisionsquote (%)")
plt.ylabel("Anzahl Bestellungen")
plt.tight_layout()
plt.show()

# Bucket-Verteilung (fixe Buckets für Reporting; optional anpassen)
bucket_bins = [-0.0001, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 1000]
bucket_labels = ["0%", "0–5%", "5–10%", "10–15%", "15–20%", "20–25%", "25–30%", "30–35%", "35–40%", "40–45%", "45–50%", ">50%"]

df["commission_rate_bucket_fixed"] = pd.cut(
    df["commission_rate_pct"],
    bins=bucket_bins,
    labels=bucket_labels
)

bucket_counts = (
    df["commission_rate_bucket_fixed"]
    .value_counts()
    .reindex(bucket_labels)
    .fillna(0)
    .astype(int)
)
bucket_pct = bucket_counts / bucket_counts.sum() * 100

plt.figure()
plt.bar(bucket_counts.index.astype(str), bucket_counts.values)
plt.title("Häufigkeit der Provisionsquote")
plt.xlabel("Provisionsquote")
plt.ylabel("Anzahl Bestellungen")
plt.xticks(rotation=15)

for i, label in enumerate(bucket_counts.index):
    v = int(bucket_counts.loc[label])
    p = bucket_pct.loc[label]
    plt.text(i, v + 3, f"{v} ({p:.1f}%)", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.show()

# ----------------------------
# 2) Provisionsquote vs durchschnittliche Bestellgröße
# ----------------------------
x = df["commission_rate_pct"]
y = df["Order Value"]
m = x.notna() & y.notna()
lr = linregress(x[m], y[m])

plt.figure()
plt.scatter(x[m], y[m], s=12)
plt.title("Bestellwert vs. Provisionsquote")
plt.xlabel("Provisionsquote")
plt.ylabel("Bestellwert (Order Value)")

x_line = np.linspace(x[m].min(), x[m].max(), 100)
y_line = lr.intercept + lr.slope * x_line
plt.plot(x_line, y_line)
plt.tight_layout()
plt.show()

print("\n=== Provisionsquote -> Bestellwert ===")
print(f"Regression: slope={lr.slope:.6f}, intercept={lr.intercept:.2f}, R²={lr.rvalue**2:.4f}")
r_p, p_p = stats.pearsonr(x[m], y[m])
r_s, p_s = stats.spearmanr(x[m], y[m])
print(f"Pearson r={r_p:.4f} (p={p_p:.4g}), Spearman ρ={r_s:.4f} (p={p_s:.4g})")

# Zusätzlich: Quartile-Buckets (gleich große Gruppen)
df["commission_rate_bucket_q"] = pd.qcut(
    df["commission_rate_pct"], 10,
    labels=["D1", "D2", "D3", "D4 ", "D5", "D6", "D7", "D8", "D9", "D10"]
)
q_order = ["D1", "D2", "D3", "D4 ", "D5", "D6", "D7", "D8", "D9", "D10"]

q_summary = (
    df.groupby("commission_rate_bucket_q", observed=True)
      .agg(
          orders=("Order ID", "count"),
          median_commission_rate_pct=("commission_rate_pct", "median"),
          avg_order_value=("Order Value", "mean"),
      )
      .reindex(q_order)
      .reset_index()
)
q_summary["median_commission_rate_pct"] = q_summary["median_commission_rate_pct"].round(2)
q_summary["avg_order_value"] = q_summary["avg_order_value"].round(2)

print("\n=== Ø Bestellwert nach Provisionsquote-Quartil ===")
print(q_summary.to_string(index=False))

plt.figure()
plt.bar(q_summary["commission_rate_bucket_q"].astype(str), q_summary["avg_order_value"].values)
plt.title("Avg Bestellwert nach Provisionsquotesezil")
plt.xlabel("Provisionsquotedezil")
plt.ylabel("Avg Bestellwert")
for i, v in enumerate(q_summary["avg_order_value"].values):
    plt.text(i, v + 5, f"{v:.0f}", ha="center", va="bottom")
plt.tight_layout()
plt.show()

# ----------------------------
# 3) Provisionsquote vs Häufigkeit der Rückerstattungen
# ----------------------------
q_refund = (
    df.groupby("commission_rate_bucket_q", observed=True)["refund_flag"]
      .agg(orders="count", refund_rate="mean")
      .reindex(q_order)
      .reset_index()
)
q_refund["refund_rate_%"] = (q_refund["refund_rate"] * 100).round(1)

print("\n=== Refund-Rate nach Provisionsquote-Quartil ===")
print(q_refund[["commission_rate_bucket_q", "orders", "refund_rate_%"]].to_string(index=False))

plt.figure()
plt.bar(q_refund["commission_rate_bucket_q"].astype(str), q_refund["refund_rate_%"].values)
plt.title("Refund-Rate nach Provisionsquotedezil")
plt.xlabel("Provisionsquotedezil")
plt.ylabel("Refund-Rate (%)")
for i, v in enumerate(q_refund["refund_rate_%"].values):
    plt.text(i, v + 0.5, f"{v:.1f}%", ha="center", va="bottom")
plt.tight_layout()
plt.show()

# Zusätzliche Tests: commission_rate_pct vs refund_flag (0/1) -> point-biserial/Pearson
rf = df["refund_flag"].astype(int)
m2 = df["commission_rate_pct"].notna() & rf.notna()
r_rf, p_rf = stats.pearsonr(df.loc[m2, "commission_rate_pct"], rf.loc[m2])
print("\n=== Provisionsquote -> Refund (ja/nein) ===")
print(f"Point-biserial/Pearson r={r_rf:.4f}, p={p_rf:.4g}")

