"""
Frage:
Beeinflusst die Höhe der Liefergebühren (Delivery Fee) die Wahl der Zahlungsart
und die Rückbuchungsrate (Refunds/Chargebacks > 0)?

Enthält:
- Kreuztabellen + Prozent
- Chi²-Tests + Cramér's V (Effektstärke)
- Visualisierungen:
  1) Gestapelte Balken (Anteil %) Zahlungsarten je Liefergebühr inkl. %-Labels
  2) Balken Refund-Rate (%) je Liefergebühr inkl. %-Labels
  3) Boxplot Liefergebühr nach Zahlungsart
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ----------------------------
# 1) Daten laden
# ----------------------------
CSV_PATH = "food_orders_new_delhi_ext2.csv"  # ggf. Pfad anpassen
df = pd.read_csv(CSV_PATH)

# ----------------------------
# 2) Aufbereitung
# ----------------------------
df["Delivery Fee"] = pd.to_numeric(df["Delivery Fee"], errors="coerce")
df["Refunds/Chargebacks"] = pd.to_numeric(df["Refunds/Chargebacks"], errors="coerce").fillna(0)
df["refund_flag"] = df["Refunds/Chargebacks"] > 0

fee_order = [0, 20, 30, 40, 50]
pm_order = ["Cash on Delivery", "Credit Card", "Digital Wallet"]

# ----------------------------
# 3) Liefergebühr -> Zahlungsart
# ----------------------------
ct_pm = pd.crosstab(df["Delivery Fee"], df["Payment Method"]).reindex(
    index=fee_order, columns=pm_order, fill_value=0
)
ct_pm_pct = ct_pm.div(ct_pm.sum(axis=1), axis=0) * 100

chi2_pm, p_pm, dof_pm, _ = stats.chi2_contingency(ct_pm)
n_pm = ct_pm.to_numpy().sum()
cramers_v_pm = np.sqrt(chi2_pm / (n_pm * (min(ct_pm.shape) - 1)))

print("\n=== Zahlungsart ~ Liefergebühr ===")
print("Anzahl:\n", ct_pm)
print("\nProzent je Liefergebühr:\n", ct_pm_pct.round(1))
print(f"Chi²={chi2_pm:.3f}, df={dof_pm}, p={p_pm:.3f}, Cramér's V={cramers_v_pm:.3f}")

# Plot 1: gestapelte Prozentbalken + %-Labels
plt.figure()
bottom = np.zeros(len(fee_order))
x = np.arange(len(fee_order))

for pm in pm_order:
    vals = ct_pm_pct[pm].values
    plt.bar(x, vals, bottom=bottom, label=pm)

    # %-Labels in die Segmente (nur wenn Segment groß genug)
    for i, v in enumerate(vals):
        if v >= 5:
            plt.text(i, bottom[i] + v / 2, f"{v:.1f}%", ha="center", va="center")

    bottom += vals

plt.title("Zahlungsartenverteilung nach Liefergebühr")
plt.xlabel("Liefergebühr")
plt.ylabel("Anteil je Zahlungsart (%)")
plt.xticks(x, [str(f) for f in fee_order])
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------
# 4) Liefergebühr -> Rückbuchungsrate
# ----------------------------
ct_ref = pd.crosstab(df["Delivery Fee"], df["refund_flag"]).reindex(index=fee_order, fill_value=0)
ct_ref = ct_ref.reindex(columns=[False, True], fill_value=0)

chi2_ref, p_ref, dof_ref, _ = stats.chi2_contingency(ct_ref)
n_ref = ct_ref.to_numpy().sum()
cramers_v_ref = np.sqrt(chi2_ref / (n_ref * (min(ct_ref.shape) - 1)))

refund_rate_pct = (ct_ref[True] / ct_ref.sum(axis=1) * 100).reindex(fee_order)

print("\n=== Refund (ja/nein) ~ Liefergebühr ===")
refund_summary = pd.DataFrame({
    "Delivery Fee": fee_order,
    "orders": ct_ref.sum(axis=1).reindex(fee_order).values,
    "refunds": ct_ref[True].reindex(fee_order).values,
    "refund_rate_%": refund_rate_pct.round(1).values
})
print(refund_summary.to_string(index=False))
print(f"Chi²={chi2_ref:.3f}, df={dof_ref}, p={p_ref:.3f}, Cramér's V={cramers_v_ref:.3f}")

# Plot 2: Refund-Rate je Liefergebühr + Label
plt.figure()
plt.bar([str(f) for f in fee_order], refund_rate_pct.values)
plt.title("Rückbuchungsrate nach Liefergebühr")
plt.xlabel("Liefergebühr")
plt.ylabel("Rückbuchungsrate (%)")

for i, f in enumerate(fee_order):
    v = refund_rate_pct.loc[f]
    plt.text(i, v + 0.5, f"{v:.1f}%", ha="center", va="bottom")

plt.tight_layout()
plt.show()

# ----------------------------
# 5) Zusatz: Liefergebühr nach Zahlungsart (Boxplot + optional Tests)
# ----------------------------
# Boxplot
plt.figure()
plt.boxplot(
    [df.loc[df["Payment Method"] == m, "Delivery Fee"].dropna() for m in pm_order],
    labels=pm_order  # kompatibel mit vielen Matplotlib-Versionen
)
plt.title("Liefergebühr nach Zahlungsart")
plt.xlabel("Zahlungsart")
plt.ylabel("Liefergebühr")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# Optional: ANOVA/Kruskal (Liefergebühr als Zahl je Zahlungsart)
groups_fee = [df.loc[df["Payment Method"] == m, "Delivery Fee"].dropna() for m in pm_order]
anova_fee = stats.f_oneway(*groups_fee)
kruskal_fee = stats.kruskal(*groups_fee)

print("\n=== Zusatztests: Delivery Fee ~ Payment Method ===")
print(f"ANOVA: F={anova_fee.statistic:.3f}, p={anova_fee.pvalue:.3f}")
print(f"Kruskal–Wallis: H={kruskal_fee.statistic:.3f}, p={kruskal_fee.pvalue:.3f}")



fee_order = [0, 20, 30, 40, 50]  # falls noch nicht gesetzt

fee_counts = (
    df["Delivery Fee"]
    .value_counts()
    .reindex(fee_order)
    .fillna(0)
    .astype(int)
)

fee_pct = fee_counts / fee_counts.sum() * 100

plt.figure()
plt.bar([str(f) for f in fee_order], fee_counts.values)
plt.title("Verteilung der Liefergebühren")
plt.xlabel("Liefergebühr")
plt.ylabel("Anzahl Bestellungen")

# Labels: Anzahl + Prozent
for i, f in enumerate(fee_order):
    v = fee_counts.loc[f]
    p = fee_pct.loc[f]
    plt.text(i, v + 3, f"{v} ({p:.1f}%)", ha="center", va="bottom")

plt.tight_layout()
plt.show()