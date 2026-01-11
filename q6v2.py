import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy import stats

CSV_PATH = "food_orders_new_delhi_ext2.csv"
df = pd.read_csv(CSV_PATH)

# --- Numeric ---
df["Order Value"] = pd.to_numeric(df["Order Value"], errors="coerce")
df["delivery_minutes"] = pd.to_timedelta(df["Delivery Duration"], errors="coerce").dt.total_seconds() / 60

# --- Rabattbetrag berechnen ---
def parse_discount(discount_str, order_value):
    """Return (discount_amount, discount_pct)"""
    if pd.isna(discount_str) or str(discount_str).strip()=="" or pd.isna(order_value) or order_value <= 0:
        return 0.0, 0.0

    s = str(discount_str).strip()

    m = re.search(r"(\d+)\s*%", s)
    if m:
        pct = float(m.group(1))
        amt = order_value * (pct / 100.0)
        return amt, pct

    m = re.search(r"(\d+)\s*off", s, flags=re.IGNORECASE)
    if m:
        amt = float(m.group(1))
        pct = (amt / order_value) * 100.0 if order_value > 0 else 0.0
        return amt, pct

    return 0.0, 0.0

tmp = df.apply(lambda r: parse_discount(r["Discounts and Offers"], r["Order Value"]), axis=1, result_type="expand")
df["discount_amount"] = tmp[0]
df["discount_rate_pct"] = tmp[1]
df["discount_flag"] = df["discount_amount"] > 0

# --- Segmente (75%-Quantile) ---
ov_q75 = df["Order Value"].quantile(0.75)
del_q75 = df["delivery_minutes"].quantile(0.75)

df["high_order"] = df["Order Value"] >= ov_q75
df["long_delivery"] = df["delivery_minutes"] >= del_q75

df["segment"] = np.select(
    [~df["high_order"] & ~df["long_delivery"],
     ~df["high_order"] &  df["long_delivery"],
      df["high_order"] & ~df["long_delivery"],
      df["high_order"] &  df["long_delivery"]],
    ["niedrig / kurz", "niedrig / lang", "hoch / kurz", "hoch / lang"],
    default="unbekannt"
)

seg_order = ["niedrig / kurz", "niedrig / lang", "hoch / kurz", "hoch / lang"]

print(f"Schwellenwerte: hoher Bestellwert >= {ov_q75:.2f}, lange Lieferzeit >= {del_q75:.1f} min")
print(df["segment"].value_counts().reindex(seg_order))

# --- Segment Summary ---
seg_summary = (
    df.groupby("segment")
      .agg(
          orders=("Order ID", "count"),
          avg_order_value=("Order Value", "mean"),
          avg_delivery_min=("delivery_minutes", "mean"),
          discount_share=("discount_flag", "mean"),
          avg_discount_amount=("discount_amount", "mean"),
          avg_discount_amount_if_discount=("discount_amount", lambda s: s[s>0].mean() if (s>0).any() else np.nan),
          median_discount_amount_if_discount=("discount_amount", lambda s: s[s>0].median() if (s>0).any() else np.nan),
      )
      .reindex(seg_order)
)

seg_summary["discount_share_%"] = seg_summary["discount_share"] * 100
seg_summary = seg_summary.drop(columns=["discount_share"]).round(2)
print("\nSegment Summary:\n", seg_summary)

# --- Tests ---
ct = pd.crosstab(df["segment"], df["discount_flag"]).reindex(index=seg_order)
ct = ct.reindex(columns=[False, True], fill_value=0)
chi2, p, dof, _ = stats.chi2_contingency(ct)
print(f"\nChi² (Rabatt ja/nein ~ Segment): p={p:.3f}")

disc_only = df[df["discount_flag"]].copy()
groups = [disc_only.loc[disc_only["segment"]==s, "discount_amount"].dropna() for s in seg_order]
kw = stats.kruskal(*groups)
print(f"Kruskal–Wallis (Rabattbetrag | nur Rabatt ~ Segment): p={kw.pvalue:.3g}")

# --- Plot 1: Rabatt-Anteil nach Segment ---
plt.figure()
share = (df.groupby("segment")["discount_flag"].mean()*100).reindex(seg_order)
plt.bar(share.index, share.values)
plt.title("Anteil Bestellungen mit Rabatt nach Segment")
plt.xlabel("Segment (Bestellwert / Lieferzeit)")
plt.ylabel("Rabatt-Anteil (%)")
plt.xticks(rotation=15)
for i, v in enumerate(share.values):
    plt.text(i, v+0.5, f"{v:.1f}%", ha="center", va="bottom")
plt.tight_layout()
plt.show()

# --- Plot 2: Ø Rabattbetrag (nur Bestellungen mit Rabatt) ---
plt.figure()
avg_amt = disc_only.groupby("segment")["discount_amount"].mean().reindex(seg_order)
plt.bar(avg_amt.index, avg_amt.values)
plt.title("Avg Rabattbetrag (Bestellungen mit Rabatt)")
plt.xlabel("Segment (Bestellwert / Lieferzeit)")
plt.ylabel("Avg Rabattbetrag")
plt.xticks(rotation=15)
for i, v in enumerate(avg_amt.values):
    plt.text(i, v+0.5, f"{v:.1f}", ha="center", va="bottom")
plt.tight_layout()
plt.show()

# --- Plot 3: Boxplot Rabattbetrag (nur Bestellungen mit Rabatt) ---
plt.figure()
data = [disc_only.loc[disc_only["segment"]==s, "discount_amount"].dropna() for s in seg_order]
try:
    plt.boxplot(data, tick_labels=seg_order)
except TypeError:
    plt.boxplot(data, labels=seg_order)
plt.title("Rabattbetrag-Verteilung (nur Bestellungen mit Rabatt)")
plt.xlabel("Segment")
plt.ylabel("Rabattbetrag")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
