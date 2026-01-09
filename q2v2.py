import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("food_orders_new_delhi_ext2.csv")

# Parse Zeit & Dauer
df["order_time_dt"] = pd.to_datetime(df["Order Time"], format="%H:%M:%S", errors="coerce")
df["order_hour"] = (df["order_time_dt"].dt.hour
                    + df["order_time_dt"].dt.minute/60
                    + df["order_time_dt"].dt.second/3600)

df["delivery_minutes"] = pd.to_timedelta(df["Delivery Duration"], errors="coerce").dt.total_seconds()/60


DAY_START, DAY_END = 6, 18
df["Tag/Nacht"] = np.where((df["order_hour"] >= DAY_START) & (df["order_hour"] < DAY_END), "Tag", "Nacht")
df["is_day"] = (df["Tag/Nacht"] == "Tag").astype(int)


print(df.groupby("Tag/Nacht")["delivery_minutes"].agg(["count","mean","median","std"]))
r, p = stats.pearsonr(df["is_day"], df["delivery_minutes"])
print(f"r={r:.4f}, p={p:.4f}")

h = df["order_hour"]
df["Tageszeit"] = np.select(
    [
        (h >= 6) & (h < 12),
        (h >= 12) & (h < 18),
        (h >= 18) & (h < 24),
        (h >= 0) | (h < 6),
    ],
    ["Morgen", "Tag", "Abend", "Nacht"],
    default="Unbekannt"
)

df = df[df["Tageszeit"] != "Unbekannt"]

# 1) Summary
print(df.groupby("Tageszeit")["delivery_minutes"].agg(["count","mean","median","std"]))

# 2) ANOVA / Kruskal
groups = [df.loc[df["Tageszeit"]==g, "delivery_minutes"].dropna()
          for g in ["Morgen","Tag","Abend","Nacht"]]
print(stats.f_oneway(*groups))      # ANOVA
print(stats.kruskal(*groups))       # Kruskal–Wallis


# Visualisierung 1: Boxplot
plt.figure()
df.boxplot(column="delivery_minutes", by="Tag/Nacht")
plt.suptitle("")
plt.title("Lieferdauer (Minuten) nach Tag/Nacht")
plt.xlabel("Tag/Nacht")
plt.ylabel("Lieferdauer (Minuten)")
plt.show()

# Visualisierung 2: Scatter (Stunde vs Dauer)
plt.figure()
plt.scatter(df["order_hour"], df["delivery_minutes"], s=12)
plt.title("Lieferdauer vs. Bestell-Uhrzeit")
plt.xlabel("Bestellzeit (Stunde 0–24)")
plt.ylabel("Lieferdauer (Minuten)")
plt.show()




# 3) Boxplot
plt.figure()
df.boxplot(column="delivery_minutes", by="Tageszeit")
plt.suptitle("")
plt.title("Lieferdauer nach Tageszeit")
plt.xlabel("Tageszeit")
plt.ylabel("Lieferdauer (Minuten)")
plt.show()
