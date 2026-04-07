import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import seaborn as sns
DATASET_NAME = "cleveland"
RESULT_DIR = os.path.join("results", DATASET_NAME)
os.makedirs(RESULT_DIR, exist_ok=True)

print("Saving all plots to:", RESULT_DIR)
df = pd.read_csv("Datasets/heart_cleveland_upload.csv")  

df['gender'] = df['sex'].map({0: 'Female', 1: 'Male'})
df['condition'] = df['condition'].map({0: 'No Disease', 1: 'Disease'})

df['group'] = df['gender'] + ' - ' + df['condition']

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
    'Male - No Disease': '#f18f01',
    'Female - No Disease': '#f18f01',
    'Male - Disease': '#009ffd',
    'Female - Disease': '#009ffd'
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
    "Cleveland Dataset - Gender Distribution",
    fontsize=18,
    fontweight='bold',
    fontname='Times New Roman'
)

ax.axis('equal')
plt.savefig(
    os.path.join(RESULT_DIR, "gender_distribution_pie.png"),
    dpi=600,
    bbox_inches='tight'
)
print("Saved: gender_distribution_pie.png")
plt.show()

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'

df = pd.read_csv("Datasets/heart_cleveland_upload.csv")

bins = [30, 40, 50, 60, 70,120]
labels = ['30-40', '40-50', '50-60', '60-70','>70']
df['age'] = pd.cut(df['age'], bins=bins, labels=labels)

df['condition'] = df['condition'].map({0: 'No Disease', 1: 'With Disease'})

grouped = df.groupby(['age', 'condition']).size().unstack(fill_value=0)

colors = ['#fae588', '#f18f01']


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


ax.set_yticks([0, 50, 100])

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

ax.set_title("Cleveland Dataset", fontsize=16, fontweight='bold')
ax.set_xlabel("Age Groups", fontsize=14, fontweight='bold')
ax.set_ylabel("Number of Patients", fontsize=14, fontweight='bold')

# Tick styling
ax.tick_params(axis='x', labelsize=12, rotation=0)
ax.tick_params(axis='y', labelsize=12)

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')


ax.grid(axis='y', linestyle='--', alpha=0.6)

legend = ax.legend(title="Condition", fontsize=11, title_fontsize=12)
for text in legend.get_texts():
    text.set_fontweight('bold')
legend.get_title().set_fontweight('bold')

plt.tight_layout()
plt.savefig(
    os.path.join(RESULT_DIR, "age_group_distribution.png"),
    dpi=600,
    bbox_inches='tight'
)
print("Saved: age_group_distribution.png")
plt.show()



df1 = pd.read_csv("Datasets/heart_cleveland_upload.csv")
df1['condition'] = df1['condition'].map({0: 'No Disease', 1: 'With Disease'})

def plot(ax, x_no, y_no, x_yes, y_yes, title):

    ax.scatter(x_no, y_no, color='#f18f01', label='No Disease', alpha=0.6)
    ax.scatter(x_yes, y_yes, color='#009ffd', label='With Disease', alpha=0.6)

    for x, y, color, style in [
        (x_no, y_no, '#f18f01', '-'),
        (x_yes, y_yes, '#009ffd', '--')
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
    
    
fig1, ax1 = plt.subplots(figsize=(6, 4), dpi=600)

plot(
    ax1,
    df1[df1['condition']=='No Disease']['age'],
    df1[df1['condition']=='No Disease']['trestbps'],
    df1[df1['condition']=='With Disease']['age'],
    df1[df1['condition']=='With Disease']['trestbps'],
    "Cleveland Dataset"
)

plt.tight_layout()
plt.savefig(
    os.path.join(RESULT_DIR, "age_bp_scatter_trend.png"),
    dpi=600,
    bbox_inches='tight'
)
print("Saved: age_bp_scatter_trend.png")
plt.show()

plt.figure(figsize=(6,4), dpi=300)


sns.countplot(x='cp', hue='condition', data=df1, palette=['#f18f01', '#009ffd'])

plt.title("Chest Pain Type vs Disease", fontsize=16, fontweight='bold')
plt.xlabel("Chest Pain Type (cp)", fontsize=14, fontweight='bold')
plt.ylabel("Count", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(
    os.path.join(RESULT_DIR, "chest_pain_vs_disease.png"),
    dpi=600,
    bbox_inches='tight'
)
print("Saved: chest_pain_vs_disease.png")
plt.show()


plt.figure(figsize=(8,6), dpi=300)

corr = df1.drop(columns=['condition','gender','group'], errors='ignore').corr()

sns.heatmap(corr, annot=True, cmap='rainbow', fmt=".2f")

plt.title("Cleveland Feature Correlation", fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig(
    os.path.join(RESULT_DIR, "correlation_heatmap.png"),
    dpi=600,
    bbox_inches='tight'
)
print("Saved: correlation_heatmap.png")
plt.show()


df = pd.read_csv("Datasets/heart_cleveland_upload.csv")

features = ['age', 'chol', 'thalach', 'trestbps']


plt.figure(figsize=(6, 4), dpi=600)

for col in features:
    plt.hist(
        df[col],
        bins=30,
        alpha=0.6,
        label=f"{col} ({df[col].min()}–{df[col].max()})"
    )

plt.title("Before Normalization", fontsize=16, fontweight='bold', fontname='Times New Roman')
plt.xlabel("Feature Value", fontsize=14, fontweight='bold', fontname='Times New Roman')
plt.ylabel("Frequency", fontsize=14, fontweight='bold', fontname='Times New Roman')

plt.xticks(fontsize=12, fontweight='bold', fontname='Times New Roman')
plt.yticks(fontsize=12, fontweight='bold', fontname='Times New Roman')

plt.legend(prop={'family': 'Times New Roman', 'weight': 'bold', 'size': 10})
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(
    os.path.join(RESULT_DIR, "before_normalization.png"),
    dpi=600,
    bbox_inches='tight'
)
print("Saved: before_normalization.png")
plt.show()

scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)

plt.figure(figsize=(6, 4), dpi=600)

for col in features:
    plt.hist(
        df_scaled[col],
        bins=30,
        alpha=0.6,
        label=f"{col} (Normalized)"
    )

plt.title("After Normalization", fontsize=16, fontweight='bold', fontname='Times New Roman')
plt.xlabel("Normalized Value (0–1)", fontsize=14, fontweight='bold', fontname='Times New Roman')
plt.ylabel("Frequency", fontsize=14, fontweight='bold', fontname='Times New Roman')

plt.xticks(fontsize=12, fontweight='bold', fontname='Times New Roman')
plt.yticks(fontsize=12, fontweight='bold', fontname='Times New Roman')

plt.legend(prop={'family': 'Times New Roman', 'weight': 'bold', 'size': 10})
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(
    os.path.join(RESULT_DIR, "after_normalization.png"),
    dpi=600,
    bbox_inches='tight'
)
print("Saved: after_normalization.png")
plt.show()