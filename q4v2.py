import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

CSV_PATH = "food_orders_new_delhi_ext2.csv"
df = pd.read_csv(CSV_PATH)

# Numerisch machen
df["Delivery Fee"] = pd.to_numeric(df["Delivery Fee"], errors="coerce")
df["Order Value"] = pd.to_numeric(df["Order Value"], errors="coerce")
df["Refunds/Chargebacks"] = pd.to_numeric(df["Refunds/Chargebacks"], errors="coerce").fillna(0)
df["refund_flag"] = df["Refunds/Chargebacks"] > 0

# Prozentuale Liefergebühr (Anteil am Bestellwert)
df = df[df["Order Value"] > 0].copy()  # Schutz gegen Division durch 0
df["delivery_fee_pct"] = df["Delivery Fee"] / df["Order Value"] * 100

print("\n=== delivery_fee_pct Summary ===")
print(df["delivery_fee_pct"].describe())

pm_order = ["Cash on Delivery", "Credit Card", "Digital Wallet"]

# -----------------------
# A) Einfluss auf Zahlungsart?
# -> Vergleich der Verteilungen delivery_fee_pct zwischen Zahlungsarten
# -----------------------
groups = [df.loc[df["Payment Method"] == m, "delivery_fee_pct"].dropna() for m in pm_order]

anova_res = stats.f_oneway(*groups)
kruskal_res = stats.kruskal(*groups)

print("\n=== Zahlungsart vs delivery_fee_pct ===")
print(f"ANOVA: F={anova_res.statistic:.3f}, p={anova_res.pvalue:.3f}")
print(f"Kruskal–Wallis: H={kruskal_res.statistic:.3f}, p={kruskal_res.pvalue:.3f}")

# Boxplot delivery_fee_pct nach Zahlungsart (kompatibel für alte/neue Matplotlib)
plt.figure()
data = [df.loc[df["Payment Method"] == m, "delivery_fee_pct"].dropna() for m in pm_order]
try:
    plt.boxplot(data, tick_labels=pm_order)   # Matplotlib >= 3.9
except TypeError:
    plt.boxplot(data, labels=pm_order)        # ältere Versionen

plt.title("Liefergebühr als % vom Bestellwert nach Zahlungsart")
plt.xlabel("Zahlungsart")
plt.ylabel("Liefergebühr (% vom Bestellwert)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# -----------------------
# B) Einfluss auf Refund-Rate?
# -> Vergleich delivery_fee_pct zwischen Refund ja/nein
# -----------------------
g0 = df.loc[~df["refund_flag"], "delivery_fee_pct"].dropna()
g1 = df.loc[df["refund_flag"], "delivery_fee_pct"].dropna()

tt = stats.ttest_ind(g0, g1, equal_var=False)             # Mittelwertvergleich (robust: Welch)
mw = stats.mannwhitneyu(g0, g1, alternative="two-sided")  # Median/Verteilung (nichtparametrisch)

print("\n=== Refund vs delivery_fee_pct ===")
print(f"Welch t-test: t={tt.statistic:.3f}, p={tt.pvalue:.3f}")
print(f"Mann–Whitney U: U={mw.statistic:.1f}, p={mw.pvalue:.3f}")

# Boxplot delivery_fee_pct nach Refund ja/nein
plt.figure()
data_ref = [g0, g1]
try:
    plt.boxplot(data_ref, tick_labels=["kein Refund", "Refund"])
except TypeError:
    plt.boxplot(data_ref, labels=["kein Refund", "Refund"])

plt.title("Liefergebühr als % vom Bestellwert vs. Rückbuchung")
plt.xlabel("Rückbuchung")
plt.ylabel("Liefergebühr (% vom Bestellwert)")
plt.tight_layout()
plt.show()

# -----------------------
# C) Verteilung der delivery_fee_pct (Histogramm)
# -----------------------
plt.figure()
plt.hist(df["delivery_fee_pct"].dropna(), bins=20)
plt.title("Liefergebühr als % vom Bestellwert")
plt.xlabel("Liefergebühr (% vom Bestellwert)")
plt.ylabel("Anzahl Bestellungen")
plt.tight_layout()
plt.show()

# -----------------------
# D) Optional: Refund-Rate nach Buckets der delivery_fee_pct (übersichtlich)
# -----------------------
df["fee_pct_bucket"] = pd.cut(
    df["delivery_fee_pct"],
    bins=[-0.0001, 0, 2.5, 5, 10, 1000],
    labels=["0%", "0–2.5%", "2.5–5%", "5–10%", ">10%"]
)

bucket = df.groupby("fee_pct_bucket")["refund_flag"].agg(orders="count", refund_rate="mean").reset_index()
bucket["refund_rate_%"] = (bucket["refund_rate"] * 100).round(1)
print("\n=== Refund-Rate nach delivery_fee_pct Buckets ===")
print(bucket[["fee_pct_bucket", "orders", "refund_rate_%"]].to_string(index=False))

plt.figure()
plt.bar(bucket["fee_pct_bucket"].astype(str), bucket["refund_rate_%"])
plt.title("Rückbuchungsrate nach Liefergebühr-%")
plt.xlabel("Liefergebühr (% vom Bestellwert)")
plt.ylabel("Rückbuchungsrate (%)")
for i, v in enumerate(bucket["refund_rate_%"].values):
    plt.text(i, v + 0.5, f"{v:.1f}%", ha="center", va="bottom")
plt.tight_layout()
plt.show()

bucket_order = ["0%", "0–2.5%", "2.5–5%", "5–10%", ">10%"]

bucket_counts = (
    df["fee_pct_bucket"]
    .value_counts()
    .reindex(bucket_order)
    .fillna(0)
    .astype(int)
)

bucket_pct = bucket_counts / bucket_counts.sum() * 100

plt.figure()
plt.bar(bucket_counts.index.astype(str), bucket_counts.values)
plt.title("Verteilung der Liefergebühr-% gruppiert")
plt.xlabel("Liefergebühr (% vom Bestellwert)")
plt.ylabel("Anzahl Bestellungen")
# Labels: Anzahl + Prozent
for i, label in enumerate(bucket_counts.index):
    v = bucket_counts.loc[label]
    p = bucket_pct.loc[label]
    plt.text(i, v + 3, f"{v} ({p:.1f}%)", ha="center", va="bottom")

plt.tight_layout()
plt.show()

pm_order = ["Cash on Delivery", "Credit Card", "Digital Wallet"]
bucket_order = ["0%", "0–2.5%", "2.5–5%", "5–10%", ">10%"]

# Kreuztabelle Bucket x Zahlungsart
ct_bucket_pm = pd.crosstab(df["fee_pct_bucket"], df["Payment Method"]).reindex(
    index=bucket_order, columns=pm_order, fill_value=0
)

# Prozent je Bucket (Zeilenprozente)
ct_bucket_pm_pct = ct_bucket_pm.div(ct_bucket_pm.sum(axis=1), axis=0) * 100

plt.figure()
bottom = np.zeros(len(bucket_order))
x = np.arange(len(bucket_order))

for pm in pm_order:
    vals = ct_bucket_pm_pct[pm].values
    plt.bar(x, vals, bottom=bottom, label=pm)

    # %-Labels in Segmente (nur wenn Segment groß genug)
    for i, v in enumerate(vals):
        if np.isnan(v) or v < 5:
            continue
        plt.text(i, bottom[i] + v / 2, f"{v:.1f}%", ha="center", va="center")

    bottom += vals

plt.title("Zahlungsarten-Verteilung nach Liefergebühr-% ")
plt.xlabel("Liefergebühr (% vom Bestellwert)")
plt.ylabel("Anteil je Zahlungsart (%)")
plt.xticks(x, bucket_order)
plt.legend()
plt.tight_layout()
plt.show()
