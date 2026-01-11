"""
Analyse: Rabatte als % (discount_rate_pct)

Frage:
Welche Rolle spielen Rabatte bei Bestellungen mit höherem Bestellwert und längeren Lieferzeiten?

Umsetzung:
- Rabatt in % aus "Discounts and Offers" extrahieren:
  * "5%/10%/15%" -> direkt
  * "50 off" -> (50 / Order Value) * 100
  * kein -> 0
- "Hoher Bestellwert" = oberstes Quartil (>= Q75)
- "Lange Lieferzeit" = oberstes Quartil (>= Q75)
- Segmente: niedrig/kurz, niedrig/lang, hoch/kurz, hoch/lang

Outputs:
- Tabelle: Segment-Übersicht (Rabatt-Anteil, Ø Rabatt%, Ø Rabatt% (nur mit Rabatt), Ø Bestellwert, Ø Lieferzeit)
- Tests: Chi² (Rabatt ja/nein ~ Segment), Kruskal (Rabatt% ~ Segment)
- Grafiken:
  1) Rabatt-Anteil (%) nach Segment
  2) Ø Rabatt (%) nur Bestellungen mit Rabatt nach Segment
  3) Boxplot: Lieferzeit (nur hohe Bestellwerte) nach Rabatt ja/nein
  + optional: Rabatt-Typen hoch/lang vs andere (Anzahl + %)
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy import stats

CSV_PATH = "food_orders_new_delhi_ext2.csv"  # ggf. Pfad anpassen
df = pd.read_csv(CSV_PATH)

# ----------------------------
# 0) Aufbereitung
# ----------------------------
df["Order Value"] = pd.to_numeric(df["Order Value"], errors="coerce")

# Lieferdauer -> Minuten
df["delivery_minutes"] = pd.to_timedelta(df["Delivery Duration"], errors="coerce").dt.total_seconds() / 60

# Rabatt in % berechnen (effektiv)
def parse_discount_rate_pct(discount_str, order_value):
    if pd.isna(discount_str):
        return 0.0
    s = str(discount_str).strip()

    # Prozent-Rabatt
    m = re.search(r"(\d+)\s*%", s)
    if m:
        return float(m.group(1))

    # fixer Betrag "50 off" -> in %
    m = re.search(r"(\d+)\s*off", s, flags=re.IGNORECASE)
    if m and pd.notna(order_value) and order_value > 0:
        amt = float(m.group(1))
        return (amt / order_value) * 100.0

    return 0.0

df["discount_rate_pct"] = df.apply(
    lambda r: parse_discount_rate_pct(r["Discounts and Offers"], r["Order Value"]),
    axis=1
)
df["discount_flag"] = df["discount_rate_pct"] > 0

# Rabatt-Typ (für optionale Auswertung)
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

    return s

df["offer_type"] = df["Discounts and Offers"].apply(normalize_offer)

# ----------------------------
# 1) Segmente definieren (Q75)
# ----------------------------
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

print(f"Schwellenwerte: hoher Bestellwert >= {ov_q75:.2f} (Q75), lange Lieferzeit >= {del_q75:.1f} min (Q75)")
print(df["segment"].value_counts().reindex(seg_order))

# ----------------------------
# 2) Segment-Zusammenfassung
# ----------------------------
seg_summary = (
    df.groupby("segment")
      .agg(
          orders=("Order ID", "count"),
          avg_order_value=("Order Value", "mean"),
          avg_delivery_min=("delivery_minutes", "mean"),
          discount_share=("discount_flag", "mean"),
          avg_discount_pct=("discount_rate_pct", "mean"),
          avg_discount_pct_if_discount=("discount_rate_pct", lambda s: s[s>0].mean() if (s>0).any() else np.nan),
      )
      .reindex(seg_order)
)

seg_summary["discount_share_%"] = seg_summary["discount_share"] * 100
seg_summary = seg_summary.drop(columns=["discount_share"]).round(2)

print("\nSegment Summary:\n", seg_summary)

# ----------------------------
# 3) Tests
# ----------------------------
# 3a) Chi²: Rabatt ja/nein ~ Segment
ct = pd.crosstab(df["segment"], df["discount_flag"]).reindex(index=seg_order)
ct = ct.reindex(columns=[False, True], fill_value=0)
chi2, p_chi, dof, _ = stats.chi2_contingency(ct)
print(f"\nChi² (Rabatt ja/nein ~ Segment): χ²={chi2:.3f}, df={dof}, p={p_chi:.3f}")

# 3b) Kruskal: Rabatt% ~ Segment (robust, da viele 0-Werte)
groups = [df.loc[df["segment"] == s, "discount_rate_pct"].dropna() for s in seg_order]
kw = stats.kruskal(*groups)
print(f"Kruskal–Wallis (Rabatt% ~ Segment): H={kw.statistic:.3f}, p={kw.pvalue:.3f}")

# ----------------------------
# 4) Grafiken
# ----------------------------

# Plot 1: Rabatt-Anteil (%) nach Segment
plt.figure()
share = (df.groupby("segment")["discount_flag"].mean() * 100).reindex(seg_order)
plt.bar(share.index, share.values)
plt.title("Anteil Bestellungen mit Rabatt nach Segment")
plt.xlabel("Segment (Bestellwert / Lieferzeit)")
plt.ylabel("Rabatt-Anteil (%)")
plt.xticks(rotation=15)
for i, v in enumerate(share.values):
    plt.text(i, v + 0.5, f"{v:.1f}%", ha="center", va="bottom")
plt.tight_layout()
plt.show()

# Plot 2: Ø Rabatt (%) nur Bestellungen mit Rabatt
plt.figure()
avg_if = df[df["discount_flag"]].groupby("segment")["discount_rate_pct"].mean().reindex(seg_order)
plt.bar(avg_if.index, avg_if.values)
plt.title("Avg Rabatt in % (Bestellungen mit Rabatt)")
plt.xlabel("Segment (Bestellwert / Lieferzeit)")
plt.ylabel("Avg Rabatt (%)")
plt.xticks(rotation=15)
for i, v in enumerate(avg_if.values):
    if pd.notna(v):
        plt.text(i, v + 0.2, f"{v:.2f}%", ha="center", va="bottom")
plt.tight_layout()
plt.show()

# Plot 3: Hoher Bestellwert – Lieferzeit nach Rabatt ja/nein (Boxplot)
high = df[df["high_order"]].copy()
data = [
    high.loc[~high["discount_flag"], "delivery_minutes"].dropna(),
    high.loc[high["discount_flag"], "delivery_minutes"].dropna()
]

plt.figure()
try:
    plt.boxplot(data, tick_labels=["kein Rabatt", "Rabatt"])
except TypeError:
    plt.boxplot(data, labels=["kein Rabatt", "Rabatt"])

plt.title("Hoher Bestellwert: Lieferzeit nach Rabatt ja/nein")
plt.xlabel("Rabatt")
plt.ylabel("Lieferzeit (Minuten)")
plt.tight_layout()
plt.show()

# ----------------------------
# 5) OPTIONAL: Rabatt-Typen in hoch/lang vs andere (Anzahl + %)
# ----------------------------
hl = df["segment"] == "hoch / lang"
ct_offer = pd.crosstab(np.where(hl, "hoch/lang", "andere"), df["offer_type"])
ct_offer_pct = ct_offer.div(ct_offer.sum(axis=1), axis=0) * 100

print("\nRabatt-Typen (Anzahl):\n", ct_offer)
print("\nRabatt-Typen (%):\n", ct_offer_pct.round(1))

seg_order = ["niedrig / kurz", "niedrig / lang", "hoch / kurz", "hoch / lang"]

seg_counts = df["segment"].value_counts().reindex(seg_order).fillna(0).astype(int)
seg_pct = seg_counts / seg_counts.sum() * 100

plt.figure()
plt.bar(seg_counts.index.astype(str), seg_counts.values)
plt.title("Verteilung der Segmente (Bestellwert / Lieferzeit)")
plt.xlabel("Segment")
plt.ylabel("Anzahl Bestellungen")
plt.xticks(rotation=15)

for i, label in enumerate(seg_counts.index):
    v = seg_counts.loc[label]
    p = seg_pct.loc[label]
    plt.text(i, v + 3, f"{v} ({p:.1f}%)", ha="center", va="bottom")

plt.tight_layout()
plt.show()