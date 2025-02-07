import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scikit_posthocs as sp
import seaborn as sns
from scipy.stats import shapiro, kruskal, mannwhitneyu

# Aktualisierte Daten mit neuen Conditionsnamen
data = {
    'Sample': ['Crop1', 'Crop2', 'Crop3', 'Crop4', 'Crop5', 'Crop6', 'Crop7', 'Crop8', 'Crop9', 'Crop10', 'Crop11', 'Crop12'],
    'Control Center': [14.525, 10.525, 18.525, 18.525, 18.525, 14.525, 8.525, None, 18.525, 12.525, 10.525, 14.525],
    'Control Periphery': [20.525, 18.525, 18.525, 20.525, None, 18.525, 18.525, 18.525, 18.525, 18.525, None, 18.525],
    'Occluder Periphery': [None, 20.525, 18.525, 20.525, 20.525, None, None, 24.525, 20.525, None, None, 20.525],
    'Control Center (2)': [14.525, 16.525, None, 18.525, None, None, 16.525, 16.525, 18.525, 18.525, 12.525, 18.525],
    'Occluder Center': [20.525, 18.525, 18.525, None, None, 14.525, 18.525, 14.525, 18.525, None, 18.525, None],
    'Occluder Periphery (2)': [20.525, 20.525, None, 20.525, 20.525, 20.525, 18.525, 20.525, 20.525, None, None, None],
    'Red Filter Center': [12.525, 14.525, 12.525, 12.525, 12.525, 12.525, 14.525, 12.525, 12.525, 12.525, 16.525, None],
    'Red Filter Periphery': [26.525, 22.525, 22.525, None, 24.525, None, None, 24.525, None, 26.525, 22.525, 24.525]
}

# In ein DataFrame umwandeln
df = pd.DataFrame(data)

# None-Werte durch np.nan ersetzen
df.replace([None], np.nan, inplace=True)

# DataFrame umformen - Beide Spalten 'Control_3Dcenter' und 'Control_3Dcenter (2)' zu einer gemeinsamen Bedingung
df_melted = pd.melt(df, id_vars="Sample", var_name="Condition", value_name="Value").dropna()

# Die Bedingungen 'Control_3Dcenter' und 'Control_3Dcenter (2)' in eine einzelne Bedingung "Control Center" umbenennen
df_melted['Condition'] = df_melted['Condition'].replace({
    'Control Center': 'Control Center',
    'Control Center (2)': 'Control Center',
    'Occluder Periphery': 'Occluder Periphery',
    'Occluder Periphery (2)': 'Occluder Periphery'
})

############################################################################## Tabelle definieren

def save_as_image(df, title, filename):
    col_widths = [max(len(str(x)) for x in df[col]) for col in df.columns]
    col_widths = [max(w, len(str(col))) * 0.15 for w, col in zip(col_widths, df.columns)]
    fig_width = sum(col_widths)
    fig_height = max(1, len(df) * 0.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('tight')
    ax.axis('off')
    ax.set_title(title, fontsize=16, pad=20)

    table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index,
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for col_idx, col_width in enumerate(col_widths):
        for row_idx in range(len(df) + 1):
            cell = table[row_idx, col_idx]
            cell.set_width(col_width)

    for key, cell in table.get_celld().items():
        cell.set_edgecolor('black')
        cell.set_linewidth(0.5)
        cell.set_text_props(ha='center', wrap=True)

    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)

############################################################################## Plots erstellen

palette = {
    'Control Center': 'lightblue',
    'Control Periphery': 'lightcoral',
    'Occluder Center': 'lightblue',
    'Occluder Periphery': 'lightcoral',
    'Red Filter Center': 'lightblue',
    'Red Filter Periphery': 'lightcoral'
}

order = [
    'Control Center',
    'Control Periphery',
    'Occluder Center',
    'Occluder Periphery',
    'Red Filter Center',
    'Red Filter Periphery'
]

fig, ax = plt.subplots(figsize=(12, 8))

sns.violinplot(
    x="Condition", y="Value", data=df_melted, palette=palette, order=order, scale="width", cut=0, inner="box", ax=ax
)

for i, condition in enumerate(order):
    y_data = df_melted[df_melted["Condition"] == condition]["Value"]
    x_data = np.random.normal(loc=i, scale=0.05, size=len(y_data))
    ax.scatter(x_data, y_data, color=palette[condition], alpha=0.6, edgecolor="black", linewidth=0.5)
    ax.text(
        i, 
        max(y_data) + (max(y_data) - min(y_data)) * 0.05 if not y_data.empty else 0,
        f'n={len(y_data)}',
        ha='center', fontsize=12
    )

ax.set_title("Raincloud Plot: OS Length per Condition (Python)", fontsize=16, weight='bold')
ax.set_xlabel("Condition", fontsize=14)
ax.set_ylabel("Values (\u00b5m)", fontsize=14)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
plt.tight_layout()
plt.show()



# Shapiro-Wilk Test für Normalverteilung
normality_results = df_melted.groupby("Condition")["Value"].apply(lambda x: shapiro(x.dropna()))
normality_summary = normality_results.apply(lambda x: {"W-statistic": x[0], "p-value": x[1]}).to_frame(name="Normality Test")

# Kruskal-Wallis Test (nicht-parametrisch)
groups = [df_melted[df_melted["Condition"] == condition]["Value"].dropna() for condition in df_melted["Condition"].unique()]
kruskal_stat, kruskal_p = kruskal(*groups)

# Mann-Whitney U Test für Center vs Center und Periphery vs Periphery
center_comparison = df_melted[df_melted["Condition"].isin(['Control Center', 'Occluder Center', 'Red Filter Center'])]
periphery_comparison = df_melted[df_melted["Condition"].isin(['Control Periphery', 'Occluder Periphery', 'Red Filter Periphery'])]

mannwhitney_center = []
mannwhitney_periphery = []

# Compare center vs center
for condition_1, condition_2 in [('Control Center', 'Occluder Center'),
                                  ('Control Center', 'Red Filter Center'),
                                  ('Occluder Center', 'Red Filter Center')]:
    group_1 = center_comparison[center_comparison["Condition"] == condition_1]['Value'].dropna()
    group_2 = center_comparison[center_comparison["Condition"] == condition_2]['Value'].dropna()

    u_stat, p_val = mannwhitneyu(group_1, group_2, alternative='two-sided')
    mannwhitney_center.append({
        'Condition_1': condition_1,
        'Condition_2': condition_2,
        'U-statistic': u_stat,
        'p-value': p_val
    })

# Compare periphery vs periphery
for condition_1, condition_2 in [('Control Periphery', 'Occluder Periphery'),
                                  ('Control Periphery', 'Red Filter Periphery'),
                                  ('Occluder Periphery', 'Red Filter Periphery')]:
    group_1 = periphery_comparison[periphery_comparison["Condition"] == condition_1]['Value'].dropna()
    group_2 = periphery_comparison[periphery_comparison["Condition"] == condition_2]['Value'].dropna()

    u_stat, p_val = mannwhitneyu(group_1, group_2, alternative='two-sided')
    mannwhitney_periphery.append({
        'Condition_1': condition_1,
        'Condition_2': condition_2,
        'U-statistic': u_stat,
        'p-value': p_val
    })

# Mann-Whitney U Test für Center vs Periphery innerhalb jeder Bedingung
center_periphery_comparison = df_melted[df_melted["Condition"].isin([
    'Control Center', 'Control Periphery',
    'Occluder Center', 'Occluder Periphery',
    'Red Filter Center', 'Red Filter Periphery'
])]

mannwhitney_center_vs_periphery = []

# Compare center vs periphery within each condition
for condition_1, condition_2 in [('Control Center', 'Control Periphery'),
                                  ('Occluder Center', 'Occluder Periphery'),
                                  ('Red Filter Center', 'Red Filter Periphery')]:
    group_1 = center_periphery_comparison[center_periphery_comparison["Condition"] == condition_1]['Value'].dropna()
    group_2 = center_periphery_comparison[center_periphery_comparison["Condition"] == condition_2]['Value'].dropna()

    u_stat, p_val = mannwhitneyu(group_1, group_2, alternative='two-sided')
    mannwhitney_center_vs_periphery.append({
        'Condition_1': condition_1,
        'Condition_2': condition_2,
        'U-statistic': u_stat,
        'p-value': p_val
    })

# Ergebnisse zusammenführen
mannwhitney_center_df = pd.DataFrame(mannwhitney_center)
mannwhitney_periphery_df = pd.DataFrame(mannwhitney_periphery)
combined_df = pd.concat([mannwhitney_center_df, 
                         mannwhitney_periphery_df,
                         pd.DataFrame(mannwhitney_center_vs_periphery)], 
                        keys=['Center vs Center', 'Periphery vs Periphery', 'Center vs Periphery within conditions'])
combined_df.reset_index(level=0, inplace=True)
combined_df.rename(columns={'level_0': 'Test Comparison'}, inplace=True)

# Dunn-Test
control_center = df_melted[df_melted["Condition"] == 'Control Center']['Value'].dropna()
occluder_center = df_melted[df_melted["Condition"] == 'Occluder Center']['Value'].dropna()
redfilter_center = df_melted[df_melted["Condition"] == 'Red Filter Center']['Value'].dropna()

control_periphery = df_melted[df_melted["Condition"] == 'Control Periphery']['Value'].dropna()
occluder_periphery = df_melted[df_melted["Condition"] == 'Occluder Periphery']['Value'].dropna()
redfilter_periphery = df_melted[df_melted["Condition"] == 'Red Filter Periphery']['Value'].dropna()

# Dunn-Test mit Bonferroni-Korrektur durchführen
res_center = sp.posthoc_dunn([control_center, occluder_center, redfilter_center], p_adjust='fdr_bh')
res_periphery = sp.posthoc_dunn([control_periphery, occluder_periphery, redfilter_periphery], p_adjust='fdr_bh')

# Ergebnisse formatieren
test_results = {
    'Comparison': ['Occluder Center vs Control', 'Red Filter Center vs Control', 'Occluder Periphery vs Control', 'Red Filter Periphery vs Control'],
    'P-value': [
        res_center.iloc[1, 0], res_center.iloc[2, 0],  # Center-Vergleiche
        res_periphery.iloc[1, 0], res_periphery.iloc[2, 0]  # Periphery-Vergleiche
    ]
}

df_results = pd.DataFrame(test_results)
df_results = df_results.round(6)


# Ergebnisse speichern

normality_summary_df = pd.DataFrame(normality_summary["Normality Test"].tolist(), index=normality_summary.index)
normality_summary_df.columns = ["W-statistic", "p-value"]
kruskal_results_df = pd.DataFrame({"Statistic": [kruskal_stat], "p-value": [kruskal_p]})
normality_summary_df = normality_summary_df.round(6)
combined_df = combined_df.round(6)

# Save normality results
save_as_image(normality_summary_df, "Shapiro-Wilk Normality Tests", "Python_normality_tests.png")

# Save Kruskal-Wallis results
save_as_image(kruskal_results_df, "Kruskal-Wallis Test", "Python_kruskal_results.png")

# Save Mann-Whitney U test
save_as_image(combined_df, "Mann-Whitney U Test Results", "Python_MannWhitney_results.png")

# Tabelle als Bild speichern
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('tight')
ax.axis('off')
ax.table(cellText=df_results.values, colLabels=df_results.columns, loc='center', cellLoc='center', colColours=['#f5f5f5']*2)

# Bild speichern
plt.savefig(r"C:\Users\gildi\OneDrive\Desktop\Masterarbeit\Data\Outer segment length\Python_dunn_test_results.png", dpi=300, bbox_inches="tight")
plt.close()