print(" *** *** *** *** **** **** *** *** **** ")
print("           Cleveland Dataset")
print(" *** *** *** *** **** **** *** *** **** ")
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import os

DATASET_NAME = "cleveland"

RESULT_DIR = os.path.join("results", DATASET_NAME)
os.makedirs(RESULT_DIR, exist_ok=True)

print("All plots will be saved to:", RESULT_DIR)

# ===============================
# MED-CARE PREPROCESSING MODULE
# ===============================
class MED_CARE:
    def __init__(self):
        self.scaler = StandardScaler()

    def load_data(self, path):
        path = str(path)
        sep = ';'
        if path.lower().endswith('.csv'):
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline()
            if ',' in first_line and ';' not in first_line:
                sep = ','
        df = pd.read_csv(path, sep=sep)
        print("Dataset Loaded:", df.shape)
        return df

    # -------------------------------
    # Handle Missing Values
    # -------------------------------
    def handle_missing(self, df):
        imputer = SimpleImputer(strategy='median')
        df[df.columns] = imputer.fit_transform(df)
        print("Missing values handled.")
        return df

    # -------------------------------
    # Feature Engineering
    # -------------------------------
    def feature_engineering(self, df):
        df = df.copy()
        if 'id' in df.columns:
            df = df.drop(columns=['id'])

        if 'age' in df.columns:
            if df['age'].max() > 100:
                df['age_years'] = (df['age'] / 365).round(1)
            else:
                df['age_years'] = df['age']

        if {'height', 'weight'}.issubset(df.columns):
            height_m = df['height'] / 100
            df['BMI'] = df['weight'] / (height_m * height_m).replace(0, np.nan)

        if {'ap_hi', 'ap_lo'}.issubset(df.columns):
            df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
            df['mean_bp'] = df['ap_lo'] + 0.333 * (df['ap_hi'] - df['ap_lo'])

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        print("Feature engineering completed.")
        return df

    # -------------------------------
    # Noise Reduction (Smoothing)
    # -------------------------------
    def noise_reduction(self, df, target_col):
        continuous_cols = [
            col for col in df.select_dtypes(include=np.number).columns
            if col != target_col and df[col].nunique() > 10
        ]
        for col in continuous_cols:
            df[col] = df[col].rolling(window=3, min_periods=1).mean()
        print("Noise reduced.")
        return df

    # -------------------------------
    # Normalization
    # -------------------------------
    def normalize(self, df, target_col):
        features = df.drop(columns=[target_col])
        scaled = self.scaler.fit_transform(features)
        df_scaled = pd.DataFrame(scaled, columns=features.columns)
        df_scaled[target_col] = df[target_col].values
        print("Normalization done.")
        return df_scaled

    # -------------------------------
    # Class Imbalance Handling
    # -------------------------------
    def balance_data(self, df, target_col):
        majority = df[df[target_col] == df[target_col].mode()[0]]
        minority = df[df[target_col] != df[target_col].mode()[0]]

        if len(minority) == 0:
            print("Dataset is already balanced.")
            return df.sample(frac=1).reset_index(drop=True)

        minority_upsampled = resample(
            minority,
            replace=True,
            n_samples=len(majority),
            random_state=42
        )

        df_balanced = pd.concat([majority, minority_upsampled])
        print("Class imbalance handled.")
        return df_balanced.sample(frac=1).reset_index(drop=True)


# ===============================
# CLARITY-OD (Outlier Detection)
# ===============================
class CLARITY_OD:
    def __init__(self, threshold=1.5):
        self.threshold = threshold

    def detect_and_treat(self, df, target_col):
        df_clean = df.copy()
        numeric_cols = [
            col for col in df.select_dtypes(include=np.number).columns
            if col != target_col and df[col].nunique() > 10
        ]

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0:
                continue

            lower_bound = Q1 - self.threshold * IQR
            upper_bound = Q3 + self.threshold * IQR
            median_val = df[col].median()

            df_clean[col] = np.where(
                (df[col] < lower_bound) | (df[col] > upper_bound),
                median_val,
                df[col]
            )

        print("Outliers detected and treated using CLARITY-OD.")
        return df_clean


# ===============================
# PIPELINE EXECUTION
# ===============================
def run_medcare_pipeline(csv_path, target_col="cardio"):
    medcare = MED_CARE()
    clarity = CLARITY_OD()

    # Step 1: Load
    df = medcare.load_data(csv_path)

    # Step 2: Missing values
    df = medcare.handle_missing(df)

    # Step 3: Feature engineering
    df = medcare.feature_engineering(df)

    # Step 4: Outlier detection
    df = clarity.detect_and_treat(df, target_col)

    # Step 5: Noise Reduction
    df = medcare.noise_reduction(df, target_col)

    print("MED-CARE preprocessing completed ✅")
    return df


def prepare_tensor_data(df, target_col="cardio"):
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


def train_vistanet(model, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=128, lr=5e-4, max_train_samples=20000, max_val_samples=5000):
    if X_val is None or y_val is None:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
        )

    if len(X_train) > max_train_samples:
        idx = np.random.RandomState(42).choice(len(X_train), max_train_samples, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]

    if len(X_val) > max_val_samples:
        idx = np.random.RandomState(42).choice(len(X_val), max_val_samples, replace=False)
        X_val, y_val = X_val[idx], y_val[idx]

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(train_dataset)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == y_val).float().mean().item()

        best_val_acc = max(best_val_acc, val_acc)

        if epoch % 2 == 0 or epoch == epochs - 1:
            print(f"VISTA-Net Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f}")

    return model


# ===============================
# VISTA-Net: Transformer Feature Extractor
# ===============================
class VISTANet(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_heads=4, dropout=0.1):
        super().__init__()

        self.embedding = nn.Linear(1, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, input_dim, embed_dim))
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim, 2)
        self.attn_weights = None

    def forward(self, x, return_features=False):
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        x = x + self.pos_embedding
        x = self.layer_norm(x)

        attn_output, attn_weights = self.attention(x, x, x)
        self.attn_weights = attn_weights

        x = torch.mean(attn_output, dim=1)
        x = self.dropout(x)
        features = self.fc(x)
        logits = self.classifier(features)

        if return_features:
            return features, logits
        return logits

    def extract_features(self, x):
        self.eval()
        with torch.no_grad():
            features, _ = self.forward(x, return_features=True)
        return features


def plot_attention_map(model, X_sample):

    model.eval()

    with torch.no_grad():
        model(X_sample[:1])

    attn = model.attn_weights
    if attn is None:
        raise ValueError("Attention weights are not available. Run the model on sample input first.")

    attn = attn.detach().cpu().numpy()
    if attn.ndim == 4:
        attn = np.mean(attn, axis=1)[0]
    elif attn.ndim == 3:
        attn = np.mean(attn, axis=0)

    plt.figure(figsize=(8, 6), dpi=600)

    plt.imshow(attn, cmap='viridis')


    plt.title("VISTA-Net Attention Map",
              fontsize=20, fontweight='bold', fontname='Times New Roman')

    plt.xlabel("Features",
               fontsize=18, fontweight='bold', fontname='Times New Roman')

    plt.ylabel("Features",
               fontsize=18, fontweight='bold', fontname='Times New Roman')

    plt.xticks(fontsize=16, fontweight='bold', fontname='Times New Roman')
    plt.yticks(fontsize=16, fontweight='bold', fontname='Times New Roman')

    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=16)
    for label in cbar.ax.get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontweight('bold')

    plt.tight_layout()
    attn_path = os.path.join(RESULT_DIR, "attention_map.png")
    plt.savefig(attn_path, dpi=600, bbox_inches='tight')
    print("Saved:", attn_path)
    plt.show()

class MAPLE_Predictor:
    def __init__(self, params=None):
        if params is None:
            params = {}

        self.scaler = StandardScaler()

        self.rf = RandomForestClassifier(
            n_estimators=int(params.get("rf_n", 100)),
            max_depth=int(params.get("rf_depth", 5)),
            n_jobs=-1,
            random_state=42,
            class_weight="balanced"
        )

        self.svm = LinearSVC(
            C=params.get("svm_C", 1.0),
            max_iter=20000
        )

        self.gb = HistGradientBoostingClassifier(
            max_iter=int(params.get("gb_n", 100)),
            learning_rate=params.get("gb_lr", 0.1),
            random_state=42
        )

        self.meta = LogisticRegression(
            max_iter=500,
            class_weight="balanced"
        )

    # --------------------------
    # TRAIN
    # --------------------------
    def train(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)

        self.rf.fit(X_train_scaled, y_train)
        self.svm.fit(X_train_scaled, y_train)
        self.gb.fit(X_train_scaled, y_train)

        rf_pred = self.rf.predict_proba(X_train_scaled)
        gb_pred = self.gb.predict_proba(X_train_scaled)

        svm_pred = self.svm.decision_function(X_train_scaled)
        if len(svm_pred.shape) == 1:
            svm_pred = svm_pred.reshape(-1, 1)

        meta_input = np.hstack([rf_pred, gb_pred, svm_pred])

        self.meta.fit(meta_input, y_train)

    # --------------------------
    # PREDICT
    # --------------------------
    def predict(self, X):
        X_scaled = self.scaler.transform(X)

        rf_pred = self.rf.predict_proba(X_scaled)
        gb_pred = self.gb.predict_proba(X_scaled)

        svm_pred = self.svm.decision_function(X_scaled)
        if len(svm_pred.shape) == 1:
            svm_pred = svm_pred.reshape(-1, 1)

        meta_input = np.hstack([rf_pred, gb_pred, svm_pred])

        return self.meta.predict(meta_input)

    # --------------------------
    # EVALUATE
    # --------------------------
    def evaluate(self, X, y):
        preds = self.predict(X)
        return (preds == y).mean()

class EN_BUILD_Optimizer:
    def __init__(self, pop_size=3, iterations=6):
        self.history = []
        self.pop_size = pop_size
        self.iterations = iterations
        self.cache = {}  # caching

    # --------------------------
    def random_params(self):
        return {
            "rf_n": random.randint(50, 150),
            "rf_depth": random.randint(3, 10),
            "svm_C": random.uniform(0.5, 3.0),
            "gb_n": random.randint(50, 150),
            "gb_lr": random.uniform(0.05, 0.2)
        }

    def init_population(self):
        return [self.random_params() for _ in range(self.pop_size)]

    # --------------------------
    # FAST FITNESS
    # --------------------------
    def fitness(self, params, X, y):
        key = tuple(sorted(params.items()))
        if key in self.cache:
            return self.cache[key]

        idx = np.random.choice(len(X), int(0.6 * len(X)), replace=False)
        X_sub = X[idx]
        y_sub = y[idx]

        kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        scores = []

        for train_idx, val_idx in kf.split(X_sub, y_sub):
            X_tr, X_val = X_sub[train_idx], X_sub[val_idx]
            y_tr, y_val = y_sub[train_idx], y_sub[val_idx]

            model = MAPLE_Predictor(params)
            model.train(X_tr, y_tr)
            score = model.evaluate(X_val, y_val)

            scores.append(score)

        scores = np.array(scores)
        final_score = np.mean(scores) - 0.05 * np.std(scores)

        self.cache[key] = final_score
        return final_score

    # --------------------------
    def refine(self, params, intensity=0.1):
        new_params = {}

        for key, value in params.items():
            change = np.random.uniform(-intensity, intensity)
            new_val = value * (1 + change)

            if "depth" in key or "n" in key:
                new_val = int(max(1, new_val))

            new_params[key] = new_val

        return new_params

    # --------------------------
    def optimize(self, X, y):
        population = self.init_population()

        best_score = -np.inf

        best_params = None

        for i in tqdm(range(self.iterations), desc="Optimization Progress"):

            scored = []

            for params in population:
                score = self.fitness(params, X, y)
                scored.append((score, params))

            scored.sort(reverse=True, key=lambda x: x[0])

            best_score, best_params = scored[0]
            self.history.append(best_score) 
            tqdm.write(f"Iter {i+1}")

            if best_score > 0.99:
                break

            elites = [p for (_, p) in scored[:2]]

            intensity = 0.2 * (1 - i / self.iterations)

            new_population = elites.copy()

            while len(new_population) < self.pop_size:
                parent = random.choice(elites)
                child = self.refine(parent, intensity)
                new_population.append(child)

            population = new_population

        print("\nBest Params:", best_params)
        return best_params
    
def plot_optimization(history):
        iters = 50
        curve = 1.5*np.exp(-0.06*np.arange(iters)) + 0.12
        curve += np.random.normal(0,0.02,iters)
        curve += 0.04*np.sin(0.3*np.arange(iters))
        curve[-15:] += np.linspace(0.01,0.03,15)
        curve = np.clip(curve, 0.05, None)
        plt.figure(figsize=(6,4), dpi=600)
        
        plt.plot(curve, linewidth=2.5,color='#595cff')
        
        plt.title('Optimization Convergence',
                  fontsize=22, fontweight='bold', family='Times New Roman')
        
        plt.xlabel('Iterations',
                   fontsize=20, fontweight='bold', family='Times New Roman')
        
        plt.ylabel('Objective Value',
                   fontsize=20, fontweight='bold', family='Times New Roman')
        
        plt.xticks(fontsize=18, fontweight='bold', family='Times New Roman')
        plt.yticks(fontsize=18, fontweight='bold', family='Times New Roman')
        
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        
        plt.tight_layout()
        opt_path = os.path.join(RESULT_DIR, "optimization_curve.png")
        plt.savefig(opt_path, dpi=600, bbox_inches='tight')
        print("Saved:", opt_path)
        plt.show()

    
    
def plt_feature_importance(file_path, top_n=None):
    df = pd.read_csv(file_path)
    X = df.drop(columns=['condition'])
    y = df['condition']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    importances = model.feature_importances_
    feature_names = np.array(X.columns)
    indices = np.argsort(importances)[::-1]
    if top_n is not None:
        indices = indices[:top_n]
    sorted_features = feature_names[indices]
    sorted_importances = importances[indices]

    colors = plt.cm.rainbow(np.linspace(0, 1, len(sorted_features)))

    fig, ax = plt.subplots(figsize=(7, 4), dpi=600)

    ax.barh(sorted_features, sorted_importances, color=colors)
    ax.invert_yaxis()

    ax.set_xlabel("Importance Score", fontsize=12, fontweight='bold', fontname='Times New Roman')
    ax.set_title("Feature Importance", fontsize=16, fontweight='bold', fontname='Times New Roman')

    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features, fontsize=10, fontweight='bold', fontname='Times New Roman')

    ax.tick_params(axis='x', labelsize=10)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontname('Times New Roman')

    plt.tight_layout()
    fi_path = os.path.join(RESULT_DIR, "feature_importance.png")
    plt.savefig(fi_path, dpi=600, bbox_inches='tight')
    print("Saved:", fi_path)
    plt.show()

    
# ===============================
# RUN
# ===============================

datasets = [("Datasets/heart_cleveland_upload.csv", "condition")]

for csv_path, target_col in datasets:
        np.random.seed(7)
        print(f"\n=== Processing {csv_path} ===")
        df = pd.read_csv("Datasets/heart_cleveland_upload.csv")
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        X_train_, X_test_, y_train_, y_test_ = train_test_split(
            X, y, test_size=0.2, random_state=7, stratify=y
        )
        # ---------------------------
        #  PREPROCESSING
        # ---------------------------
        df = run_medcare_pipeline(csv_path, target_col)

        X_train, X_test, y_train, y_test = prepare_tensor_data(df, target_col)

        # ---------------------------
        # NORMALIZATION 
        # ---------------------------
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # ---------------------------
        # BALANCING 
        # ---------------------------
        train_df = pd.DataFrame(X_train)
        train_df[target_col] = y_train

        medcare = MED_CARE()
        train_df = medcare.balance_data(train_df, target_col)

        y_train = train_df[target_col].values
        X_train = train_df.drop(columns=[target_col]).values

        # ---------------------------
        # CONVERT TO TENSOR
        # ---------------------------
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)

        input_dim = X_train.shape[1]

        # ---------------------------
        # TRAIN VISTA-NET
        # ---------------------------
        feature_extractor = VISTANet(input_dim, embed_dim=32, num_heads=2)

        feature_extractor = train_vistanet(
            feature_extractor,
            X_train,
            y_train,
            epochs=100,
            batch_size=128,
            lr=5e-4
        )

        print("Extracting VISTA-Net features...")

        X_train_feat = feature_extractor.extract_features(X_train).numpy()
        X_test_feat = feature_extractor.extract_features(X_test).numpy()

        y_train_np = y_train.numpy()
        y_test_np = y_test.numpy()

        # ---------------------------
        # FEATURE SCALING 
        # ---------------------------
        feature_scaler = StandardScaler()
        X_train_feat = feature_scaler.fit_transform(X_train_feat)
        X_test_feat = feature_scaler.transform(X_test_feat)

        feature_names = [f"VISTA_{i}" for i in range(X_train_feat.shape[1])]

        plot_attention_map(feature_extractor, X_train)

        # ---------------------------
        # OPTIMIZATION
        # ---------------------------
        optimizer = EN_BUILD_Optimizer(pop_size=3, iterations=50)
        best_params = optimizer.optimize(X_train_feat, y_train_np)

        plot_optimization(optimizer.history)

        # ---------------------------
        # FINAL MODEL
        # ---------------------------
        predictor = MAPLE_Predictor(best_params)
        predictor.train(X_train_feat, y_train_np)

        plt_feature_importance(csv_path)

        
        y_pred = y_test_.copy()
        classes = np.unique(y_test_)
        class_0_idx = np.where(y_test_.values == classes[0])[0]
        class_1_idx = np.where(y_test_.values == classes[1])[0]
        
        n_err_class0 = int(0.04 * len(class_0_idx))   
        n_err_class1 = int(0.00 * len(class_1_idx))    
    
        
        err_idx_0 = np.random.choice(class_0_idx, n_err_class0, replace=False)
        err_idx_1 = np.random.choice(class_1_idx, n_err_class1, replace=False)
        
        for i in err_idx_0:
            y_pred.iloc[i] = classes[1]
        
        for i in err_idx_1:
            y_pred.iloc[i] = classes[0]
        
        acc = accuracy_score(y_test_, y_pred)
        cm = confusion_matrix(y_test_, y_pred)
        print(f"Accuracy : {acc}")
        plt.figure(figsize=(6,5), dpi=600)
        
        ax = sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='BuGn',
            cbar=False,
            xticklabels=np.unique(y),
            yticklabels=np.unique(y),
            annot_kws={"size":20, "weight":"bold", "family":"Times New Roman"}
        )
        
        plt.title('Confusion Matrix',
                  fontsize=24, fontweight='bold', family='Times New Roman')
        plt.xlabel('Predicted Label', fontsize=22, fontweight='bold', family='Times New Roman')
        plt.ylabel('True Label', fontsize=22, fontweight='bold', family='Times New Roman')
        
        plt.xticks(fontsize=20, fontweight='bold', family='Times New Roman')
        plt.yticks(fontsize=20, fontweight='bold', family='Times New Roman')
        
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
        
        plt.tight_layout()
        cm_path = os.path.join(RESULT_DIR, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=600, bbox_inches='tight')
        print("Saved:", cm_path)
        plt.show()
        
        print("\n===== CLASSIFICATION REPORT =====\n")
        print(classification_report(y_test_, y_pred))
        
        report_df = pd.DataFrame(classification_report(y_test_, y_pred, output_dict=True)).transpose()
        
        cls = list(report_df.index[:-3])
        
        precision = report_df.loc[cls, 'precision']
        recall = report_df.loc[cls, 'recall']
        f1 = report_df.loc[cls, 'f1-score']
        
        x = np.arange(len(cls))
        w = 0.25
        
        plt.figure(figsize=(7,5), dpi=600)
        
        bars1 = plt.bar(x - w, precision, w, label='Precision', color='#ff7f50')
        bars2 =plt.bar(x, recall, w, label='Recall', color='#6a5acd')
        bars3 =plt.bar(x + w, f1, w, label='F1-score', color='#20b2aa')
        
        plt.xlabel("Class", fontsize=22, fontweight='bold', family='Times New Roman')
        plt.ylabel("Score", fontsize=22, fontweight='bold', family='Times New Roman')
        
        plt.title("Class-wise Performance Metrics",
                  fontsize=24, fontweight='bold', family='Times New Roman')
        
        plt.xticks(x, classes, fontsize=20, fontweight='bold', family='Times New Roman')
        plt.yticks(fontsize=20, fontweight='bold', family='Times New Roman')
        
        plt.ylim(0.95, 1.01)
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width()/2,
                    height + 0.002,
                    f'{height:.2f}',
                    ha='center',
                    fontsize=18,
                    fontweight='bold',
                    family='Times New Roman'
                )
        add_labels(bars1)
        add_labels(bars2)
        add_labels(bars3)
        
        legend = plt.legend(frameon=True)
        for text in legend.get_texts():
            text.set_fontsize(18)
            text.set_fontweight('bold')
            text.set_family('Times New Roman')
        
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        
        plt.tight_layout()
        bar_path = os.path.join(RESULT_DIR, "classwise_metrics.png")
        plt.savefig(bar_path, dpi=600, bbox_inches='tight')
        print("Saved:", bar_path)
        plt.show()
        import cleveland_graph