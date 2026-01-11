import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import linregress

CSV_PATH = "food_orders_new_delhi_ext2.csv"  # Pfad anpassen
df = pd.read_csv(CSV_PATH)

# --- Spalten prüfen ---
required = ["Order Value", "Commission Fee", "Payment Processing Fee"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Fehlende Spalten: {missing}. Verfügbar: {list(df.columns)}")

# --- Numerisch machen ---
for col in required:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["Total Fees"] = df["Commission Fee"] + df["Payment Processing Fee"]

# --- Korrelationen (Pearson + Spearman) ---
def corr_report(x, y, label):
    m = x.notna() & y.notna()
    r_p, p_p = stats.pearsonr(x[m], y[m])
    r_s, p_s = stats.spearmanr(x[m], y[m])
    print(f"{label}: n={m.sum()}, Pearson r={r_p:.4f} (p={p_p:.4g}), Spearman ρ={r_s:.4f} (p={p_s:.4g})")

print("\n=== Korrelationen: Order Value vs Gebühren ===")
corr_report(df["Order Value"], df["Commission Fee"], "Commission Fee")
corr_report(df["Order Value"], df["Payment Processing Fee"], "Payment Processing Fee")
corr_report(df["Order Value"], df["Total Fees"], "Total Fees")

# --- Regression + Scatterplot mit Linie ---
def scatter_with_regression(x, y, title, xlabel, ylabel):
    m = x.notna() & y.notna()
    x2, y2 = x[m], y[m]
    lr = linregress(x2, y2)

    plt.figure()
    plt.scatter(x2, y2, s=12)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    x_line = np.linspace(x2.min(), x2.max(), 100)
    y_line = lr.intercept + lr.slope * x_line
    plt.plot(x_line, y_line)
    plt.tight_layout()
    plt.show()

    print(f"{title}: slope={lr.slope:.6f}, intercept={lr.intercept:.2f}, R²={lr.rvalue**2:.4f}")

print("\n=== Plots + Regression ===")
scatter_with_regression(
    df["Order Value"], df["Commission Fee"],
    "Order Value vs. Commission Fee",
    "Order Value", "Commission Fee"
)

scatter_with_regression(
    df["Order Value"], df["Payment Processing Fee"],
    "Order Value vs. Payment Processing Fee",
    "Order Value", "Payment Processing Fee"
)

scatter_with_regression(
    df["Order Value"], df["Total Fees"],
    "Order Value vs. (Commission + Processing)",
    "Order Value", "Overall Fees"
)

# --- Optional: Gebühren als Anteil vom Order Value (hilft oft mehr als absolute Gebühren) ---
df["Commission_rate"] = df["Commission Fee"] / df["Order Value"]
df["Processing_rate"] = df["Payment Processing Fee"] / df["Order Value"]

print("\n=== Gebührenquoten (Anteil am Order Value) ===")
print(df[["Commission_rate","Processing_rate"]].describe(percentiles=[0.05,0.5,0.95]).T)

plt.figure()
plt.scatter(df["Order Value"], df["Commission_rate"], s=12)
plt.title("Order Value vs. Provisionsquote (Commission Fee / Order Value)")
plt.xlabel("Order Value")
plt.ylabel("Provisionsquote")
plt.tight_layout()
plt.show()

plt.figure()
plt.scatter(df["Order Value"], df["Processing_rate"], s=12)
plt.title("Order Value vs. Processing-Quote (Processing Fee / Order Value)")
plt.xlabel("Order Value")
plt.ylabel("Processing-Quote")
plt.tight_layout()
plt.show()

