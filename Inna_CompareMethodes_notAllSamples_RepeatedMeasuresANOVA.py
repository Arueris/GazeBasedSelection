import numpy as np
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt


# EyeCounting Datensatz (Goldstandard)
central_control_data_1 = [
    [45061.7284, 40000, 38765.4321],          # Chicken02
    [35308.64198, 37530.8642, 38518.51852],   # Chicken03
    [53209.87654, 59876.54321, 42716.04938],  # Chicken04
    [43086.41975, 44938.2716, 41975.30864],   # Chicken05
    [None],  # Chicken06
    [None],  # Chicken07
    [47901.23457]               # Chicken08 
]

peripheral_control_data_1 = [
    [31481.48148, 39012.34568, 30617.28395],  # Chicken02
    [36543.20988, 27160.49383, 28024.69136],  # Chicken03
    [51234.5679, 48395.06173, 51111.11111],   # Chicken04
    [31975.30864, 34074.07407, 34074.07407],  # Chicken05
    [28395.06173, 33827.16049, 35679.01235],  # Chicken06
    [40617.28395, 42345.67901, 37160.49383]   # Chicken07
]

central_treatment_data_1 = [
    [52222.22222, 46790.12346],  # Chicken02
    [35925.92593, 36172.83951, 43580.24691],  # Chicken03
    [45308.64198, 55802.46914, 50370.37037]   # Chicken04
]

peripheral_treatment_data_1 = [
    [34074.07407, 32469.1358],                # Chicken02
    [28271.60494],   # Chicken03
    [53703.7037, 35925.92593, 50740.74074]    # Chicken04
]

central_treatment2_data_1 = [
    [54444.44444, 38395.06173, 53209.87654],  # Chicken05
    [50000, 47283.95062],        # Chicken06
    [40617.28395, 41358.02469, 40740.74074],  # Chicken07
    [35679.01235, 37530.8642, 37407.40741]    # Chicken08
]

peripheral_treatment2_data_1 = [
    [43209.87654, 38024.69136, 37530.8642],   # Chicken05
    [39259.25926, 40617.28395, 40123.45679],  # Chicken06
    [44444.44444, 40987.65432, 40617.28395],  # Chicken07
    [37777.77778, 37777.77778]   # Chicken08
]

# ImageJ Datensatz (Methode 1)
central_control_data_2 = [
    [39629.62963, 40864.19753, 38271.60494],  # Chicken02
    [39259.25926, 42222.22222, 45679.01235],  # Chicken03
    [49876.54321, 55308.64198, 45185.18519],  # Chicken04
    [45555.55556, 44074.07407, 42716.04938],  # Chicken05
    [None],   # Chicken06
    [None],  # Chicken07
    [54567.90123]               # Chicken08 
]

peripheral_control_data_2 = [
    [31358.02469, 34938.2716, 31234.5679],    # Chicken02
    [32716.04938, 25308.64198, 29012.34568],  # Chicken03
    [45432.09877, 46543.20988, 47407.40741],  # Chicken04
    [35679.01235, 38148.14815, 40493.82716],  # Chicken05
    [33086.41975, 32839.50617, 35679.01235],  # Chicken06
    [42098.76543, 42098.76543, 36790.12346]   # Chicken07
]

central_treatment_data_2 = [
    [52345.67901, 46419.75309],  # Chicken02
    [35679.01235, 40123.45679, 42962.96296],  # Chicken03
    [50123.45679, 53456.79012, 47901.23457]   # Chicken04
]

peripheral_treatment_data_2 = [
    [35185.18519, 33333.33333],               # Chicken02
    [28271.60494],  # Chicken03
    [50987.65432, 40493.82716, 44691.35802]   # Chicken04
]

central_treatment2_data_2 = [
    [50370.37037, 40987.65432, 45308.64198],  # Chicken05
    [41604.93827, 39382.71605],  # Chicken06
    [47407.40741, 46172.83951, 51728.39506],  # Chicken07
    [37407.40741, 37654.32099, 40493.82716]   # Chicken08
]

peripheral_treatment2_data_2 = [
    [41234.5679, 33703.7037, 34691.35802],    # Chicken05
    [42716.04938, 38518.51852, 39753.08642],  # Chicken06
    [45925.92593, 42469.1358, 43827.16049],   # Chicken07
    [34814.81481, 36790.12346]   # Chicken08
]

# Python Datensatz (Methode 2)
central_control_data_3 = [
    [40987.6543, 45925.92593, 42839.50617],  # Chicken02
    [34938.2716, 35679.01235, 40246.91358],  # Chicken03
    [41234.5679, 47777.77778, 44444.44444],  # Chicken04
    [41728.39506, 39506.17284, 35925.92593],  # Chicken05
    [None],  # Chicken06
    [None],  # Chicken07
    [48765.4321]                # Chicken08 
]

peripheral_control_data_3 = [
    [28148.14815, 30370.37037, 32222.22222],  # Chicken02
    [36666.66667, 25802.46914, 24567.90123],  # Chicken03
    [51728.39506, 48395.06173, 50246.91358],  # Chicken04
    [39135.80247, 29259.25926, 38765.4321],   # Chicken05
    [33209.87654, 22345.67901, 44567.90123],  # Chicken06
    [33703.7037, 43456.79012, 34691.35802]    # Chicken07
]

central_treatment_data_3 = [
    [45061.7284, 44691.35802],   # Chicken02
    [35802.46914, 35185.18519, 40864.19753],  # Chicken03
    [42962.96296, 52962.96296, 41481.48148]   # Chicken04
]

peripheral_treatment_data_3 = [
    [26296.2963, 24938.2716],                 # Chicken02
    [23456.79012],  # Chicken03
    [57037.03704, 37037.03704, 50370.37037]   # Chicken04
]

central_treatment2_data_3 = [
    [50000, 40987.65432, 47901.23457],        # Chicken05
    [51975.30864, 36543.20988],  # Chicken06
    [40246.91358, 44197.53086, 60000],        # Chicken07
    [36913.58025, 32469.1358, 36296.2963]     # Chicken08
]

peripheral_treatment2_data_3 = [
    [33333.33333, 36049.38272, 30864.19753],  # Chicken05
    [48148.14815, 47407.40741, 46790.12346],  # Chicken06
    [33580.24691, 44814.81481, 47407.40741],  # Chicken07
    [35308.64198, 27037.03704]   # Chicken08
]

# Funktion zur Bereinigung der Daten
def clean_data(data):
    return [np.array([np.nan if x is None else x for x in sublist], dtype=float) for sublist in data]

# Funktion zur Erstellung eines DataFrames
def create_dataframe(data, condition, method):
    df = pd.DataFrame(data)
    df['Chicken'] = [f'Chicken{i+2}' for i in range(len(data))]
    df = df.melt(id_vars=['Chicken'], var_name='Measurement', value_name='Value')
    df['Condition'] = condition
    df['Method'] = method
    return df.dropna()

# Daten für jede Methode und Bedingung erstellen
data_eyecounting = pd.concat([
    create_dataframe(central_control_data_1, 'Central Control', 'EyeCounting'),
    create_dataframe(peripheral_control_data_1, 'Peripheral Control', 'EyeCounting'),
    create_dataframe(central_treatment_data_1, 'Central Treatment', 'EyeCounting'),
    create_dataframe(peripheral_treatment_data_1, 'Peripheral Treatment', 'EyeCounting'),
    create_dataframe(central_treatment2_data_1, 'Central Treatment 2', 'EyeCounting'),
    create_dataframe(peripheral_treatment2_data_1, 'Peripheral Treatment 2', 'EyeCounting')
])

data_imagej = pd.concat([
    create_dataframe(central_control_data_2, 'Central Control', 'ImageJ'),
    create_dataframe(peripheral_control_data_2, 'Peripheral Control', 'ImageJ'),
    create_dataframe(central_treatment_data_2, 'Central Treatment', 'ImageJ'),
    create_dataframe(peripheral_treatment_data_2, 'Peripheral Treatment', 'ImageJ'),
    create_dataframe(central_treatment2_data_2, 'Central Treatment 2', 'ImageJ'),
    create_dataframe(peripheral_treatment2_data_2, 'Peripheral Treatment 2', 'ImageJ')
])

data_python = pd.concat([
    create_dataframe(central_control_data_3, 'Central Control', 'Python'),
    create_dataframe(peripheral_control_data_3, 'Peripheral Control', 'Python'),
    create_dataframe(central_treatment_data_3, 'Central Treatment', 'Python'),
    create_dataframe(peripheral_treatment_data_3, 'Peripheral Treatment', 'Python'),
    create_dataframe(central_treatment2_data_3, 'Central Treatment 2', 'Python'),
    create_dataframe(peripheral_treatment2_data_3, 'Peripheral Treatment 2', 'Python')
])

# Alle Daten kombinieren
all_data = pd.concat([data_eyecounting, data_imagej, data_python])

# Shapiro-Wilk-Test für jede Methode
shapiro_results = []
methods = all_data['Method'].unique()

for method in methods:
    subset = all_data[all_data['Method'] == method]['Value']
    
    # Shapiro-Wilk-Test mit pingouin
    test_result = pg.normality(subset, method='shapiro')

    # Extraktion der relevanten Werte
    w_stat = test_result['W'].values[0]  # Teststatistik
    p_value = test_result['pval'].values[0]  # p-Wert
    
    # Ergebnisse speichern
    shapiro_results.append([method, round(w_stat, 4), round(p_value, 4)])

# Umwandlung der Ergebnisse in einen DataFrame
shapiro_df = pd.DataFrame(shapiro_results, columns=['Method', 'W-Value', 'p-Value'])
shapiro_shapiro_df = shapiro_df.round(4)


# Repeated Measures ANOVA durchführen
rm_anova = pg.rm_anova(data=all_data, dv='Value', within=['Method'], subject='Chicken', detailed=True)
rm_anova = rm_anova.round(4)

# Spaltennamen anpassen
rm_anova.rename(columns={
    'Source': 'Source',
    'SS': 'Sum of Squares (SS)',
    'DF': 'Degrees of Freedom (DF)',
    'MS': 'Mean Square (MS)',
    'F': 'F-Value',
    'p-unc': 'p-Value',
    'ng2': 'Partial Eta Squared (η²)',
    'eps': 'Epsilon'
}, inplace=True)

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



# Tukey's HSD durchführen
tukey_results = pg.pairwise_tukey(data=all_data, dv='Value', between='Method')

# Ergebnisse umbenennen und beschriften
tukey_results.rename(columns={
    'A': 'Method A',
    'B': 'Method B',
    'mean(A)': 'Mean A',
    'mean(B)': 'Mean B',
    'diff': 'Difference',
    'se': 'Standard Error',
    'T': 'T-value',
    'p-tukey': 'p-value (Tukey)',
    'hedges': 'Hedges g'
}, inplace=True)

# Rundung der Tukey-Ergebnisse auf 4 Nachkommastellen
tukey_results = tukey_results.round(4)

save_as_image(rm_anova, "Repeated Measures ANOVA Results (Methods)", "rm_anova_results_methods.png")

# Speichern der Shapiro-Ergebnisse als Bild
save_as_image(shapiro_df, "Shapiro-Wilk test of Normality.", "shapiro_test_results.png")


# Speichern der Tukey-Ergebnisse als Bild
save_as_image(tukey_results, "Tukey's HSD (pairwise comparisons)", "tukey_hsd_results.png")