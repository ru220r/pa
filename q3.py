import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("food_orders_new_delhi_ext2.csv")
df["Order Value"] = pd.to_numeric(df["Order Value"], errors="coerce")

def parse_discount_rate_pct(discount_str, order_value):
    if pd.isna(discount_str):
        return 0.0

    s = str(discount_str).strip()

    m = re.search(r"(\d+)\s*%", s)
    if m:
        return float(m.group(1))

    m = re.search(r"(\d+)\s*off", s, flags=re.IGNORECASE)
    if m and pd.notna(order_value) and order_value > 0:
        amt = float(m.group(1))
        return (amt / order_value) * 100.0

    return 0.0

df["discount_rate_pct"] = df.apply(
    lambda r: parse_discount_rate_pct(r["Discounts and Offers"], r["Order Value"]),
    axis=1
)

df["discount_bucket"] = pd.cut(
    df["discount_rate_pct"],
    bins=[-0.0001, 0.0001, 5, 10, 15, 1000],
    labels=["kein", "bis 5%", ">5–10%", ">10–15%", ">15%"]
)

ct = pd.crosstab(df["discount_bucket"], df["Payment Method"])
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
print("\nAnzahl:\n", ct)
print("\nProzent je Bucket:\n", ct_pct.round(1))


chi2, p, dof, _ = stats.chi2_contingency(ct)
n = ct.to_numpy().sum()
k = min(ct.shape)
cramers_v = np.sqrt(chi2 / (n * (k - 1)))
print(f"\nChi²={chi2:.3f}, df={dof}, p={p:.3f}")
print(f"Cramér's V={cramers_v:.3f}")

order_buckets = ["kein", "bis 5%", ">5–10%", ">10–15%", ">15%"]
ct_pct_plot = ct_pct.reindex(order_buckets)

plt.figure()

bottom = np.zeros(len(ct_pct_plot))
x_labels = ct_pct_plot.index.astype(str)

for pm in ct_pct_plot.columns:
    vals = ct_pct_plot[pm].values
    plt.bar(x_labels, vals, bottom=bottom, label=pm)

    for i, v in enumerate(vals):
        if np.isnan(v) or v <= 0:
            continue
        y = bottom[i] + v / 2  # Mitte des Segments
        plt.text(i, y, f"{v:.1f}%", ha="center", va="center")

    bottom += vals

plt.title("Zahlungsartenverteilung nach Rabattgröße)")
plt.xlabel("Rabattgröße (% vom Bestellwert)")
plt.ylabel("Anteil je Zahlungsart (%)")
plt.legend()
plt.tight_layout()
plt.show()

methods = df["Payment Method"].dropna().unique()
data = [df.loc[df["Payment Method"] == m, "discount_rate_pct"].dropna() for m in methods]

plt.figure()
plt.boxplot(data, tick_labels=methods)
plt.title("Rabatt (%) nach Zahlungsart")
plt.xlabel("Zahlungsart")
plt.ylabel("Rabatt (% vom Bestellwert)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

def normalize_offer(s):
    if pd.isna(s):
        return "kein"
    s = str(s).strip()

    m = re.search(r"(\d+)\s*%", s)
    if m:
        return f"{m.group(1)}%"

    m = re.search(r"(\d+)\s*off", s, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1)} off"

    return s  # Fallback

df["offer_type"] = df["Discounts and Offers"].apply(normalize_offer)

wanted_order = ["kein", "5%", "10%", "15%", "50 off"]

offer_counts = df["offer_type"].value_counts()
offer_counts = offer_counts.reindex(wanted_order, fill_value=0)

offer_pct = offer_counts / offer_counts.sum() * 100

plt.figure()
plt.bar(offer_counts.index, offer_counts.values)
plt.title("Häufigkeit der Rabatttypen")
plt.xlabel("Rabatttyp")
plt.ylabel("Anzahl Bestellungen")
plt.xticks(rotation=0)

for i, label in enumerate(offer_counts.index):
    v = int(offer_counts.loc[label])
    p = offer_pct.loc[label]
    plt.text(i, v + 3, f"{v} ({p:.1f}%)", ha="center", va="bottom")

plt.tight_layout()
plt.show()

bucket_order = ["kein", "bis 5%", ">5–10%", ">10–15%", ">15%"]

bucket_counts = (
    df["discount_bucket"]
    .value_counts()
    .reindex(bucket_order)
    .fillna(0)
    .astype(int)
)

bucket_pct = bucket_counts / bucket_counts.sum() * 100

plt.figure()
plt.bar(bucket_counts.index.astype(str), bucket_counts.values)
plt.title("Häufigkeit der Rabattgrößen")
plt.xlabel("Rabattgröße (% vom Bestellwert)")
plt.ylabel("Anzahl Bestellungen")

# Labels oben: Anzahl + Prozent
for i, label in enumerate(bucket_counts.index):
    v = bucket_counts.loc[label]
    p = bucket_pct.loc[label]
    plt.text(i, v + 3, f"{v} ({p:.1f}%)", ha="center", va="bottom")

plt.tight_layout()
plt.show()
