import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'

DATASET_NAME = "Cardiovascular"  

RESULT_DIR = os.path.join("results", DATASET_NAME)
os.makedirs(RESULT_DIR, exist_ok=True)

print("Saving results in:", RESULT_DIR)
df = pd.read_csv("Datasets/Cardiovascular Heart Disease Dataset.csv", sep=';')

df['gender'] = df['gender'].map({1: 'Female', 2: 'Male'})
df['cardio'] = df['cardio'].map({0: 'No Disease', 1: 'Disease'})

df['group'] = df['gender'] + ' - ' + df['cardio']

counts = df['group'].value_counts()
print(counts)

order = [
    'Male - No Disease',
    'Male - Disease',
    'Female - No Disease',
    'Female - Disease'
]


counts = counts.reindex(order, fill_value=0)

color_map = {
    'Male - No Disease': '#66b3ff',
    'Female - No Disease': '#66b3ff',
    'Male - Disease': '#cc2936',
    'Female - Disease': '#cc2936'
}

colors = [color_map[label] for label in counts.index]

fig, ax = plt.subplots(figsize=(5, 5))

wedges, texts, autotexts = ax.pie(
    counts,
    labels=counts.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=colors
)

for text in texts:
    text.set_fontsize(14)
    text.set_fontweight('bold')
    text.set_fontname('Times New Roman')

for autotext in autotexts:
    autotext.set_fontsize(13)
    autotext.set_fontweight('bold')
    autotext.set_fontname('Times New Roman')
    autotext.set_color('black')

ax.set_title(
    "Cardiovascular Dataset - Gender Distribution",
    fontsize=18,
    fontweight='bold',
    fontname='Times New Roman'
)

ax.axis('equal')
pie_path = os.path.join(RESULT_DIR, "gender_distribution_pie.png")
plt.savefig(pie_path, dpi=600, bbox_inches='tight')
print("Saved:", pie_path)
plt.show()

df = pd.read_csv("Datasets/Cardiovascular Heart Disease Dataset.csv", sep=';')

df['age_years'] = df['age'] / 365

bins = [30, 40, 50, 60, 70]
labels = ['30-40', '40-50', '50-60', '60-70']
df['age_group'] = pd.cut(df['age_years'], bins=bins, labels=labels)

df['cardio'] = df['cardio'].map({0: 'No Disease', 1: 'With Disease'})

grouped = df.groupby(['age_group', 'cardio']).size().unstack(fill_value=0)
colors = ['#d8a7a0', '#c0392b']  

fig, ax = plt.subplots(figsize=(6, 3), dpi=300)

grouped.plot(
    kind='bar',
    stacked=True,
    color=colors,
    edgecolor='black',
    linewidth=0.8,
    ax=ax
)
total = grouped.sum(axis=1)

y_max = total.max()
ax.set_ylim(0, y_max* 1.25)  


ax.set_yticks([0,10000,20000,30000])
for i, (idx, row) in enumerate(grouped.iterrows()):
    cumulative = row.sum()
    
    percent = (row.get('With Disease', 0) / total[idx]) * 100

    ax.text(
        i,
        cumulative + y_max * 0.05,  
        f"{percent:.1f}%",
        ha='center',
        fontsize=12,
        fontweight='bold'
    )

plt.title("Cardiovascular Dataset", fontsize=16, fontweight='bold')
ax.set_xlabel("Age Groups", fontsize=14, fontweight='bold')
ax.set_ylabel("Number of Patients", fontsize=14, fontweight='bold')

ax.tick_params(axis='x', labelsize=12, rotation=0)
ax.tick_params(axis='y', labelsize=12)
plt.xticks(rotation=0)
legend = ax.legend(title="Condition", fontsize=11, title_fontsize=12)
for text in legend.get_texts():
    text.set_fontweight('bold')
legend.get_title().set_fontweight('bold')
plt.tight_layout()
bar_path = os.path.join(RESULT_DIR, "age_group_distribution.png")
plt.savefig(bar_path, dpi=600, bbox_inches='tight')
print("Saved:", bar_path)
plt.show()


#-------------------------------

df2 = pd.read_csv("Datasets/Cardiovascular Heart Disease Dataset.csv", sep=';')
df2['condition'] = df2['cardio'].map({0: 'No Disease', 1: 'With Disease'})
df2['age_years'] = df2['age'] / 365

def plot(ax, x_no, y_no, x_yes, y_yes, title):

    ax.scatter(x_no, y_no, color='#1f77b4', label='No Disease', alpha=0.6)
    ax.scatter(x_yes, y_yes, color='#d62728', label='With Disease', alpha=0.6)

    for x, y, color, style in [
        (x_no, y_no, '#1f77b4', '-'),
        (x_yes, y_yes, '#d62728', '--')
    ]:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), linestyle=style, color=color, linewidth=2)

    # Titles and labels
    ax.set_title(title, fontsize=18, fontweight='bold', fontname='Times New Roman')
    ax.set_xlabel("Age", fontsize=16, fontweight='bold', fontname='Times New Roman')
    ax.set_ylabel("Blood Pressure", fontsize=16, fontweight='bold', fontname='Times New Roman')

    # Tick styling
    ax.tick_params(axis='both', labelsize=14)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontweight('bold')

    # Legend styling
    legend = ax.legend()
    for text in legend.get_texts():
        text.set_fontname('Times New Roman')
        text.set_fontweight('bold')
        text.set_fontsize(12)

    ax.grid(True, linestyle='--', alpha=0.5)
    
    
fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=600)

plot(
    ax2,
    df2[df2['condition']=='No Disease']['age_years'],
    df2[df2['condition']=='No Disease']['ap_hi'],
    df2[df2['condition']=='With Disease']['age_years'],
    df2[df2['condition']=='With Disease']['ap_hi'],
    "Cardiovascular Dataset"
)

plt.tight_layout()
scatter_path = os.path.join(RESULT_DIR, "age_bp_relationship.png")
plt.savefig(scatter_path, dpi=600, bbox_inches='tight')
print("Saved:", scatter_path)
plt.show()    


plt.figure(figsize=(6,4), dpi=300)

sns.countplot(x='gluc', hue='cardio', data=df, palette=['#66b3ff','#cc2936'])

plt.title("Glucose vs Disease", fontsize=16, fontweight='bold')
plt.xlabel("Glucose Level", fontsize=14, fontweight='bold')

plt.tight_layout()
glucose_path = os.path.join(RESULT_DIR, "glucose_vs_disease.png")
plt.savefig(glucose_path, dpi=600, bbox_inches='tight')
print("Saved:", glucose_path)
plt.show()

df_corr = pd.read_csv("Datasets/Cardiovascular Heart Disease Dataset.csv", sep=';')

df_corr['age_years'] = df_corr['age'] / 365

df_corr['bmi'] = df_corr['weight'] / ((df_corr['height']/100) ** 2)

df_corr = df_corr.drop(columns=['id'], errors='ignore')

corr = df_corr.corr()

# Plot
plt.figure(figsize=(8,6), dpi=300)

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    linewidths=0.5,
    cbar=True
)

plt.title(
    "Cardiovascular Correlation Heatmap",
    fontsize=18,
    fontweight='bold',
    fontname='Times New Roman'
)

plt.xticks(fontsize=10, rotation=45)
plt.yticks(fontsize=10)

plt.tight_layout()
heatmap_path = os.path.join(RESULT_DIR, "correlation_heatmap.png")
plt.savefig(heatmap_path, dpi=600, bbox_inches='tight')
print("Saved:", heatmap_path)
plt.show()