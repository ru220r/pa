import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


CSV_PATH = "food_orders_new_delhi_ext2.csv"  

df = pd.read_csv(CSV_PATH)

required_cols = ["Payment Method", "Order Value", "Refunds/Chargebacks", "Order ID"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Fehlende Spalten im CSV: {missing}. Verfügbare Spalten: {list(df.columns)}")

df["refund_flag"] = df["Refunds/Chargebacks"].fillna(0) > 0


summary = (
    df.groupby("Payment Method")
      .agg(
          orders=("Order ID", "count"),
          avg_order_value=("Order Value", "mean"),
          median_order_value=("Order Value", "median"),
          std_order_value=("Order Value", "std"),
          refund_count=("refund_flag", "sum"),
          refund_rate=("refund_flag", "mean"),
          avg_refund_amount_if_refund=("Refunds/Chargebacks",
                                       lambda s: s[s > 0].mean() if (s > 0).any() else np.nan),
      )
      .reset_index()
)

summary_display = summary.copy()
summary_display["avg_order_value"] = summary_display["avg_order_value"].round(2)
summary_display["median_order_value"] = summary_display["median_order_value"].round(2)
summary_display["std_order_value"] = summary_display["std_order_value"].round(2)
summary_display["refund_rate_%"] = (summary_display["refund_rate"] * 100).round(1)
summary_display["avg_refund_amount_if_refund"] = summary_display["avg_refund_amount_if_refund"].round(2)
summary_display = summary_display.drop(columns=["refund_rate"])

print("\n=== Kennzahlen nach Zahlungsart ===")
print(summary_display.to_string(index=False))


methods = df["Payment Method"].dropna().unique()
groups_order_value = [
    df.loc[df["Payment Method"] == m, "Order Value"].dropna()
    for m in methods
]

anova_res = stats.f_oneway(*groups_order_value)

cont_table = pd.crosstab(df["Payment Method"], df["refund_flag"])
chi2_stat, chi2_p, dof, expected = stats.chi2_contingency(cont_table)

print("\n=== Tests ===")
print(f"ANOVA (Bestellwert ~ Zahlungsart): F={anova_res.statistic:.3f}, p={anova_res.pvalue:.3f}")
print(f"Chi-Quadrat (Refund ja/nein ~ Zahlungsart): χ²={chi2_stat:.3f}, df={dof}, p={chi2_p:.3f}")


order = sorted(methods)

plt.figure()
df.boxplot(column="Order Value", by="Payment Method")
plt.suptitle("")
plt.title("Bestellwert nach Zahlungsart")
plt.xlabel("Zahlungsart")
plt.ylabel("Bestellwert")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

refund_rate_pct = (
    df.groupby("Payment Method")["refund_flag"]
      .mean()
      .reindex(order)
      .mul(100)
)

plt.figure()
plt.bar(refund_rate_pct.index, refund_rate_pct.values)
plt.title("Rückerstattungsrate nach Zahlungsart")
plt.xlabel("Zahlungsart")
plt.ylabel("Rückerstattungsrate (%)")
plt.xticks(rotation=15)

for i, v in enumerate(refund_rate_pct.values):
    plt.text(i, v + 0.4, f"{v:.1f}%", ha="center", va="bottom")

plt.tight_layout()
plt.show()

counts = pd.crosstab(df["Payment Method"], df["refund_flag"]).reindex(index=order)
counts = counts.reindex(columns=[False, True], fill_value=0)

plt.figure()
bars_no = plt.bar(counts.index, counts[False].values, label="Kein Refund")
bars_yes = plt.bar(counts.index, counts[True].values, bottom=counts[False].values, label="Refund")

plt.title("Refund-Häufigkeit nach Zahlungsart")
plt.xlabel("Zahlungsart")
plt.ylabel("Anzahl Bestellungen")
plt.xticks(rotation=15)

totals = counts.sum(axis=1).replace(0, np.nan)
pct_no  = counts[False] / totals * 100
pct_yes = counts[True]  / totals * 100

for i, method in enumerate(counts.index):
    no_h = counts.loc[method, False]
    yes_h = counts.loc[method, True]
    total = totals.loc[method]

    
    if no_h > 0:
        y = no_h / 2
        plt.text(i, y, f"{pct_no.loc[method]:.1f}%", ha="center", va="center")

    if yes_h > 0:
        y = no_h + yes_h / 2
        plt.text(i, y, f"{pct_yes.loc[method]:.1f}%", ha="center", va="center")

    plt.text(i, no_h + yes_h + 2, f"n={int(total)}", ha="center", va="bottom")

plt.legend()
plt.tight_layout()
plt.show()

mean_order_value = (
    df.groupby("Payment Method")["Order Value"]
      .mean()
      .reindex(order)
)

plt.figure()
plt.bar(mean_order_value.index, mean_order_value.values)
plt.title("Durchschnittlicher Bestellwert nach Zahlungsart")
plt.xlabel("Zahlungsart")
plt.ylabel("Ø Bestellwert")

for i, v in enumerate(mean_order_value.values):
    plt.text(i, v + (0.01 * mean_order_value.max()), f"{v:.2f}", ha="center", va="bottom")

plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
