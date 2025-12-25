###   G-StageNet Adaptive α learned end-to-end per CV fold + Summary
import os
import cv2
import time
import joblib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tempfile
import shutil

from glob import glob
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE

# ---------------------- CONFIG ----------------------
IMG_SIZE = 224
EMBED_DIM = 128
DATA_DIR = "./data/PAPILA"
N_SPLITS = 5
N_CLASSES = 3
FUSION_DIM = 128  # common latent dimension
CLASS_NAMES = ["Healthy", "Glaucoma", "Suspect"]
FIXED_ALPHA_VALUE = 0.5  # fixed alpha ablation

# ---------------------- Helper: robust model size computation (improved) ----------------------
def get_model_size_mb(keras_model, base_name="temp_model"):
    debug = []
    tmp_root = tempfile.mkdtemp(prefix=f"{base_name}_")
    try:
        saved_paths = []
        dir_try = os.path.join(tmp_root, base_name + "_dir")
        try:
            keras_model.save(dir_try)
            debug.append(("saved_dir", dir_try))
            saved_paths.append(dir_try)
        except Exception as e:
            debug.append(("save_dir_exception", str(e)))
        keras_file = os.path.join(tmp_root, base_name + ".keras")
        try:
            keras_model.save(keras_file)
            debug.append(("saved_keras", keras_file))
            saved_paths.append(keras_file)
        except Exception as e:
            debug.append(("save_keras_exception", str(e)))
        savedmodel_dir = os.path.join(tmp_root, base_name + "_savedmodel")
        try:
            tf.saved_model.save(keras_model, savedmodel_dir)
            debug.append(("saved_savedmodel", savedmodel_dir))
            saved_paths.append(savedmodel_dir)
        except Exception as e:
            debug.append(("savedmodel_exception", str(e)))
        weights_path = os.path.join(tmp_root, base_name + ".weights.h5")
        weights_size_mb = 0.0
        try:
            keras_model.save_weights(weights_path)
            weights_size_mb = os.path.getsize(weights_path) / 1e6
            debug.append((weights_path, os.path.getsize(weights_path)))
        except Exception as e:
            debug.append(("weights_save_exception", str(e)))
        total_bytes = 0
        detailed = []
        for p in saved_paths:
            if os.path.exists(p):
                if os.path.isfile(p):
                    try:
                        sz = os.path.getsize(p)
                    except OSError:
                        sz = 0
                    total_bytes += sz
                    detailed.append((p, sz))
                else:
                    dir_total = 0
                    for dirpath, dirnames, filenames in os.walk(p):
                        for f in filenames:
                            fp = os.path.join(dirpath, f)
                            try:
                                sz = os.path.getsize(fp)
                            except OSError:
                                sz = 0
                            dir_total += sz
                            detailed.append((fp, sz))
                    total_bytes += dir_total
        dir_size_mb = total_bytes / 1e6
        reported_mb = max(dir_size_mb, weights_size_mb)
        debug.extend(detailed)
        debug.append(("dir_size_mb", dir_size_mb))
        debug.append(("weights_size_mb", weights_size_mb))
        return reported_mb, debug
    finally:
        try:
            shutil.rmtree(tmp_root)
        except Exception:
            pass

# ---------------------- DATA LOADING ----------------------
def load_images_and_features(data_dir):
    image_paths = glob(os.path.join(data_dir, '*/*.png'))
    X_imgs, X_feats, y_labels = [], [], []
    class_map = {'healthy': 0, 'glaucoma': 1, 'glaucoma_suspect': 2}

    for path in image_paths:
        img = cv2.imread(path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X_imgs.append(img / 255.0)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)
            roundness = (4 * area) / (np.pi * (max(w, h) ** 2))
            compactness = (perimeter ** 2) / (4 * np.pi * area + 1e-6)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area != 0 else 0
        else:
            area = perimeter = aspect_ratio = circularity = roundness = compactness = solidity = 0

        mean_r = np.mean(img[..., 0])
        mean_g = np.mean(img[..., 1])
        mean_b = np.mean(img[..., 2])

        X_feats.append([area, perimeter, aspect_ratio, circularity,
                        roundness, compactness, solidity, mean_r, mean_g, mean_b])

        label = os.path.basename(os.path.dirname(path)).lower()
        y_labels.append(class_map[label])

    return np.array(X_imgs, dtype=np.float32), np.array(X_feats), np.array(y_labels)

# ---------------------- TRIPLET CNN ----------------------
def embedding_cnn(fine_tune=False):
    base = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = bool(fine_tune)
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(EMBED_DIM)(x)
    return tf.keras.models.Model(base.input, x)

def triplet_loss(_, y_pred):
    anchor, positive, negative = y_pred[:, :EMBED_DIM], y_pred[:, EMBED_DIM:2*EMBED_DIM], y_pred[:, 2*EMBED_DIM:]
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    return tf.reduce_mean(tf.maximum(pos_dist - neg_dist + 1.0, 0.0))

def build_triplet_network(fine_tune=False):
    model = embedding_cnn(fine_tune=fine_tune)
    a = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    p = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    n = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    emb_a, emb_p, emb_n = model(a), model(p), model(n)
    merged = tf.keras.layers.Concatenate()([emb_a, emb_p, emb_n])
    triplet = tf.keras.models.Model(inputs=[a, p, n], outputs=merged)
    triplet.compile(loss=triplet_loss, optimizer='adam')
    return triplet, model

def generate_triplets_dataset(images, labels, batch_size=32):
    def _generator():
        while True:
            a_imgs, p_imgs, n_imgs = [], [], []
            for _ in range(batch_size):
                idx = np.random.choice(len(images))
                anchor, anchor_label = images[idx], labels[idx]
                pos_idx = np.random.choice(np.where(labels == anchor_label)[0])
                while pos_idx == idx:
                    pos_idx = np.random.choice(np.where(labels == anchor_label)[0])
                neg_idx = np.random.choice(np.where(labels != anchor_label)[0])
                a_imgs.append(anchor)
                p_imgs.append(images[pos_idx])
                n_imgs.append(images[neg_idx])
            yield ((np.array(a_imgs), np.array(p_imgs), np.array(n_imgs)), np.ones((batch_size, 1)))
    sig = ((tf.TensorSpec((None, IMG_SIZE, IMG_SIZE, 3), tf.float32),) * 3, tf.TensorSpec((None, 1), tf.float32))
    return tf.data.Dataset.from_generator(_generator, output_signature=sig)

# ---------------------- ADAPTIVE & FIXED FUSION NETS ----------------------
class AdaptiveAlpha(tf.keras.layers.Layer):
    def __init__(self, init_value=0.5, fusion_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.init_value = init_value
        self.fusion_dim = fusion_dim

    def build(self, input_shape):
        self.alpha = self.add_weight(
            name="alpha",
            shape=(),
            initializer=tf.keras.initializers.Constant(self.init_value),
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        batch = tf.shape(inputs)[0]
        alpha_clipped = tf.clip_by_value(self.alpha, 0.0, 1.0)
        if self.fusion_dim is None:
            return tf.fill([batch, 1], alpha_clipped)
        else:
            return tf.fill([batch, self.fusion_dim], alpha_clipped)


def build_fusion_classifier(emb_dim, hand_dim, n_classes=N_CLASSES, fusion_dim=FUSION_DIM):
    emb_in = tf.keras.Input(shape=(emb_dim,), name="emb_in")
    hand_in = tf.keras.Input(shape=(hand_dim,), name="hand_in")

    proj_emb = tf.keras.layers.Dense(fusion_dim, use_bias=False, name="proj_emb")(emb_in)
    proj_hand = tf.keras.layers.Dense(fusion_dim, use_bias=False, name="proj_hand")(hand_in)

    proj_emb = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="norm_emb")(proj_emb)
    proj_hand = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="norm_hand")(proj_hand)

    alpha_tensor = AdaptiveAlpha(name="alpha_gate", fusion_dim=fusion_dim)(emb_in)

    fused = tf.keras.layers.Lambda(lambda xs: xs[2] * xs[0] + (1 - xs[2]) * xs[1],
                                   name="fused")([proj_emb, proj_hand, alpha_tensor])

    logits = tf.keras.layers.Dense(n_classes, activation='softmax', name="cls")(fused)

    model = tf.keras.Model(inputs=[emb_in, hand_in], outputs=logits, name="FusionNet")
    return model


def build_fusion_classifier_fixed_alpha(emb_dim, hand_dim, alpha_value=FIXED_ALPHA_VALUE, n_classes=N_CLASSES, fusion_dim=FUSION_DIM):
    emb_in = tf.keras.Input(shape=(emb_dim,), name="emb_in")
    hand_in = tf.keras.Input(shape=(hand_dim,), name="hand_in")

    proj_emb = tf.keras.layers.Dense(fusion_dim, use_bias=False, name="proj_emb_fixed")(emb_in)
    proj_hand = tf.keras.layers.Dense(fusion_dim, use_bias=False, name="proj_hand_fixed")(hand_in)

    proj_emb = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="norm_emb_fixed")(proj_emb)
    proj_hand = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="norm_hand_fixed")(proj_hand)

    # fixed alpha tensor using Lambda (not trainable)
    def make_alpha(x, a=alpha_value, fd=fusion_dim):
        return tf.fill([tf.shape(x)[0], fd], tf.cast(a, tf.float32))

    alpha_tensor = tf.keras.layers.Lambda(make_alpha, name="alpha_fixed")(emb_in)

    fused = tf.keras.layers.Lambda(lambda xs: xs[2] * xs[0] + (1 - xs[2]) * xs[1],
                                   name="fused_fixed")([proj_emb, proj_hand, alpha_tensor])

    logits = tf.keras.layers.Dense(n_classes, activation='softmax', name="cls_fixed")(fused)

    model = tf.keras.Model(inputs=[emb_in, hand_in], outputs=logits, name="FusionNetFixed")
    return model

# ---------------------- ALPHA TRACKER CALLBACK ----------------------
class AlphaTracker(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.alphas = []

    def on_epoch_end(self, epoch, logs=None):
        try:
            layer = self.model.get_layer("alpha_gate")
            alpha_val = float(tf.clip_by_value(layer.alpha, 0.0, 1.0).numpy())
        except Exception:
            alpha_val = None
        self.alphas.append(alpha_val)

# ---------------------- PLOTTING HELPERS ----------------------
def plot_confusion(cm, title):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(title)
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.show()

# feature correlation (raw) heatmap
def plot_raw_correlations(X, title, max_vars_to_plot=200):
    # X: numpy array (n_samples, n_features)
    df = pd.DataFrame(X)
    cols = df.columns
    if len(cols) > max_vars_to_plot:
        # show only top variables by variance to keep heatmap readable
        var_idx = np.argsort(df.var().values)[-max_vars_to_plot:]
        df = df.iloc[:, var_idx]
    corr = df.corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, vmax=1.0, vmin=-1.0, cmap='coolwarm')
    plt.title(title)
    plt.show()

# ---------------------- KERAS MLP FOR TRAIN/VAL CURVES (ablation curves) ----------------------
def build_mlp_classifier(input_dim, n_classes=N_CLASSES):
    inp = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(256, activation='relu')(inp)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_keras_classifier_get_history(X, y, name, epochs=30, batch_size=32, random_state=42):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)
    model = build_mlp_classifier(X.shape[1])
    history = model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0)

    y_pred = np.argmax(model.predict(X_te), axis=1)
    cm = confusion_matrix(y_te, y_pred)

    try:
        model.save(f"{name}_mlp.keras")
    except Exception:
        model.save_weights(f"{name}_mlp.weights.h5")

    return history, (y_te, y_pred, cm)

# ---------------------- SKLEARN EVAL (no per-fold confusion plots) ----------------------
def evaluate_model(data, labels, name):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    accs, f1s, aucs, precs, recs, times = [], [], [], [], [], []

    for train_idx, test_idx in skf.split(data, labels):
        X_train, X_test = data[train_idx], data[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        clf = VotingClassifier(estimators=[
            ('xgb', XGBClassifier(n_estimators=200, learning_rate=0.03,
                                  objective='multi:softprob', use_label_encoder=False, num_class=N_CLASSES)),
            ('lgbm', LGBMClassifier(n_estimators=200, learning_rate=0.03, objective='multiclass')),
            ('lr', LogisticRegression(max_iter=1000))
        ], voting='soft', weights=[2, 2, 1])

        start = time.time()
        clf.fit(X_train, y_train)
        duration = time.time() - start
        times.append(duration / len(X_test))

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, average='macro'))
        precs.append(precision_score(y_test, y_pred, average='macro'))
        recs.append(recall_score(y_test, y_pred, average='macro'))
        aucs.append(roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro'))

    final_model = clf.fit(data, labels)
    joblib.dump(final_model, f"{name}_final_model.pkl")
    size_mb = os.path.getsize(f"{name}_final_model.pkl") / 1e6

    print(f"\nModel: {name}")
    print(f"Avg Accuracy: {np.mean(accs):.4f}, F1: {np.mean(f1s):.4f}, AUC: {np.mean(aucs):.4f}, Time/sample: {np.mean(times):.4f}s, Size: {size_mb:.2f}MB")

    return [name, np.mean(accs), np.mean(precs), np.mean(recs), np.mean(f1s), np.mean(aucs), size_mb]

# ---------------------- FUSION EVAL (ADAPTIVE) ----------------------
def evaluate_adaptive_fusion(embeddings, feats_scaled, labels, name="GStageNet-Adaptive"):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    accs, f1s, aucs, precs, recs, times = [], [], [], [], [], []
    alphas_final = []
    alpha_histories = []
    sizes = []
    fold_histories = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(embeddings, labels)):
        Xe_tr, Xe_te = embeddings[train_idx], embeddings[test_idx]
        Xh_tr, Xh_te = feats_scaled[train_idx], feats_scaled[test_idx]
        y_tr, y_te  = labels[train_idx], labels[test_idx]

        fusion_model = build_fusion_classifier(Xe_tr.shape[1], Xh_tr.shape[1], fusion_dim=FUSION_DIM)
        fusion_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                             loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        alpha_cb = AlphaTracker()
        history = fusion_model.fit([Xe_tr, Xh_tr], y_tr, epochs=20, batch_size=64,
                         verbose=0, validation_split=0.1, callbacks=[alpha_cb])

        fold_histories.append(history.history)
        alpha_histories.append(alpha_cb.alphas)
        alpha_fold = alpha_cb.alphas[-1] if len(alpha_cb.alphas) > 0 else float(tf.clip_by_value(fusion_model.get_layer("alpha_gate").alpha, 0.0, 1.0).numpy())
        alphas_final.append(float(alpha_fold))

        fused_layer = fusion_model.get_layer("fused").output
        feat_extractor = tf.keras.Model(inputs=fusion_model.inputs, outputs=fused_layer)
        F_tr = feat_extractor.predict([Xe_tr, Xh_tr], verbose=0)
        F_te = feat_extractor.predict([Xe_te, Xh_te], verbose=0)

        clf = VotingClassifier(estimators=[
            ('xgb', XGBClassifier(n_estimators=200, learning_rate=0.03, objective='multi:softprob', use_label_encoder=False, num_class=N_CLASSES)),
            ('lgbm', LGBMClassifier(n_estimators=200, learning_rate=0.03, objective='multiclass')),
            ('lr', LogisticRegression(max_iter=1000))
        ], voting='soft', weights=[2, 2, 1])

        start = time.time()
        clf.fit(F_tr, y_tr)
        duration = time.time() - start
        times.append(duration / len(F_te))

        y_pred  = clf.predict(F_te)
        y_proba = clf.predict_proba(F_te)

        accs.append(accuracy_score(y_te, y_pred))
        f1s.append(f1_score(y_te, y_pred, average='macro'))
        precs.append(precision_score(y_te, y_pred, average='macro'))
        recs.append(recall_score(y_te, y_pred, average='macro'))
        aucs.append(roc_auc_score(y_te, y_proba, multi_class='ovr', average='macro'))

        fold_clf_path = f"{name}_ensemble_fold{fold+1}.pkl"
        joblib.dump(clf, fold_clf_path)
        sizes.append(os.path.getsize(fold_clf_path) / 1e6)

    fusion_model_full = build_fusion_classifier(embeddings.shape[1], feats_scaled.shape[1], fusion_dim=FUSION_DIM)
    fusion_model_full.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    alpha_cb_full = AlphaTracker()
    history_full = fusion_model_full.fit([embeddings, feats_scaled], labels, epochs=30, batch_size=64, verbose=0, callbacks=[alpha_cb_full])

    size_mb, debug_list = get_model_size_mb(fusion_model_full, base_name=f"{name}_final_model")

    alpha_global = alpha_cb_full.alphas[-1] if len(alpha_cb_full.alphas) > 0 else float(fusion_model_full.get_layer("alpha_gate").alpha.numpy())

    print(f"\nModel: {name} (Adaptive Feature-Level Fusion)")
    print(f"Avg Accuracy: {np.mean(accs):.4f}, F1: {np.mean(f1s):.4f}, AUC: {np.mean(aucs):.4f}, Time/sample: {np.mean(times):.4f}s")
    print(f"Learned α per fold: {['{:.2f}'.format(a) for a in alphas_final]}")
    print(f"Mean α: {np.mean(alphas_final):.4f}, Global α (all-data): {alpha_global:.4f}")
    print(f"Avg Ensemble Size (folds): {np.mean(sizes):.2f} MB, FusionModel Size (all-data): {size_mb:.2f} MB")

    summary_row = [name, np.mean(accs), np.mean(precs), np.mean(recs),
                   np.mean(f1s), np.mean(aucs), size_mb]

    return summary_row, alphas_final, fusion_model_full, alpha_histories, fold_histories, history_full.history

# ---------------------- FUSION EVAL (FIXED ALPHA) ----------------------
def evaluate_fixed_alpha_fusion(embeddings, feats_scaled, labels, alpha_value=FIXED_ALPHA_VALUE, name="GStageNet-Fixed"):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    accs, f1s, aucs, precs, recs, times = [], [], [], [], [], []
    sizes = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(embeddings, labels)):
        Xe_tr, Xe_te = embeddings[train_idx], embeddings[test_idx]
        Xh_tr, Xh_te = feats_scaled[train_idx], feats_scaled[test_idx]
        y_tr, y_te  = labels[train_idx], labels[test_idx]

        fusion_model = build_fusion_classifier_fixed_alpha(Xe_tr.shape[1], Xh_tr.shape[1], alpha_value=alpha_value, fusion_dim=FUSION_DIM)
        fusion_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        history = fusion_model.fit([Xe_tr, Xh_tr], y_tr, epochs=20, batch_size=64, verbose=0, validation_split=0.1)

        fused_layer = fusion_model.get_layer("fused_fixed").output
        feat_extractor = tf.keras.Model(inputs=fusion_model.inputs, outputs=fused_layer)
        F_tr = feat_extractor.predict([Xe_tr, Xh_tr], verbose=0)
        F_te = feat_extractor.predict([Xe_te, Xh_te], verbose=0)

        clf = VotingClassifier(estimators=[
            ('xgb', XGBClassifier(n_estimators=200, learning_rate=0.03, objective='multi:softprob', use_label_encoder=False, num_class=N_CLASSES)),
            ('lgbm', LGBMClassifier(n_estimators=200, learning_rate=0.03, objective='multiclass')),
            ('lr', LogisticRegression(max_iter=1000))
        ], voting='soft', weights=[2, 2, 1])

        start = time.time()
        clf.fit(F_tr, y_tr)
        duration = time.time() - start
        times.append(duration / len(F_te))

        y_pred  = clf.predict(F_te)
        y_proba = clf.predict_proba(F_te)

        accs.append(accuracy_score(y_te, y_pred))
        f1s.append(f1_score(y_te, y_pred, average='macro'))
        precs.append(precision_score(y_te, y_pred, average='macro'))
        recs.append(recall_score(y_te, y_pred, average='macro'))
        aucs.append(roc_auc_score(y_te, y_proba, multi_class='ovr', average='macro'))

        fold_clf_path = f"{name}_ensemble_fold{fold+1}.pkl"
        joblib.dump(clf, fold_clf_path)
        sizes.append(os.path.getsize(fold_clf_path) / 1e6)

    # train on all data to get final fused features for visualization
    fusion_model_full = build_fusion_classifier_fixed_alpha(embeddings.shape[1], feats_scaled.shape[1], alpha_value=alpha_value, fusion_dim=FUSION_DIM)
    fusion_model_full.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history_full = fusion_model_full.fit([embeddings, feats_scaled], labels, epochs=30, batch_size=64, verbose=0)

    size_mb, debug_list = get_model_size_mb(fusion_model_full, base_name=f"{name}_final_model")

    print(f"\nModel: {name} (Fixed α={alpha_value})")
    print(f"Avg Accuracy: {np.mean(accs):.4f}, F1: {np.mean(f1s):.4f}, AUC: {np.mean(aucs):.4f}, Time/sample: {np.mean(times):.4f}s")
    print(f"Avg Ensemble Size (folds): {np.mean(sizes):.2f} MB, FusionModel Size (all-data): {size_mb:.2f} MB")

    summary_row = [name, np.mean(accs), np.mean(precs), np.mean(recs), np.mean(f1s), np.mean(aucs), size_mb]
    return summary_row, fusion_model_full, history_full.history

# ---------------------- RUN PIPELINE ----------------------
if __name__ == '__main__':
    X_imgs, X_feats, y_labels = load_images_and_features(DATA_DIR)
    feats_scaled = StandardScaler().fit_transform(X_feats)

    # Train triplet embedding
    triplet_model, embed_model = build_triplet_network(fine_tune=False)
    print("Training triplet embedding model...")
    triplet_model.fit(generate_triplets_dataset(X_imgs, y_labels), steps_per_epoch=50, epochs=20)

    embeddings = embed_model.predict(X_imgs, batch_size=32)

    summary = []

    # Evaluate Handcrafted and CNN
    summary.append(evaluate_model(feats_scaled, y_labels, "Handcrafted"))
    summary.append(evaluate_model(embeddings, y_labels, "CNN"))

    # Train Keras MLPs for Handcrafted & CNN
    print("\nTraining Keras MLP (Handcrafted features) for train/val curves and held-out confusion matrix...")
    hist_handcrafted, (y_te_hand, y_pred_hand, cm_hand) = train_keras_classifier_get_history(feats_scaled, y_labels, "Handcrafted", epochs=30)

    print("\nTraining Keras MLP (CNN embeddings) for train/val curves and held-out confusion matrix...")
    hist_cnn, (y_te_cnn, y_pred_cnn, cm_cnn) = train_keras_classifier_get_history(embeddings, y_labels, "CNN", epochs=30)

    # Adaptive fusion
    fusion_results, alphas, fusion_model_full, alpha_histories, fold_histories, fusion_history_full = evaluate_adaptive_fusion(embeddings, feats_scaled, y_labels, "GStageNet")
    summary.append(fusion_results)

    # Fixed-alpha fusion
    fixed_results, fusion_model_fixed_full, fusion_fixed_history = evaluate_fixed_alpha_fusion(embeddings, feats_scaled, y_labels, alpha_value=FIXED_ALPHA_VALUE, name="GStageNet-Fixed")
    summary.append(fixed_results)

    # Obtain fused features from both fusion models (adaptive & fixed)
    feat_extractor_adaptive = tf.keras.Model(inputs=fusion_model_full.inputs, outputs=fusion_model_full.get_layer("fused").output)
    fused_adaptive_all = feat_extractor_adaptive.predict([embeddings, feats_scaled], verbose=0)

    feat_extractor_fixed = tf.keras.Model(inputs=fusion_model_fixed_full.inputs, outputs=fusion_model_fixed_full.get_layer("fused_fixed").output)
    fused_fixed_all = feat_extractor_fixed.predict([embeddings, feats_scaled], verbose=0)

    # Train Keras MLP on fused features to get held-out confusion matrix for both adaptive and fixed
    print("\nTraining Keras MLP (Fused-Adaptive features) for train/val curves and held-out confusion matrix...")
    hist_fused_adaptive, (y_te_fused_ad, y_pred_fused_ad, cm_fused_ad) = train_keras_classifier_get_history(fused_adaptive_all, y_labels, "Fused-Adaptive", epochs=30)

    print("\nTraining Keras MLP (Fused-Fixed features) for train/val curves and held-out confusion matrix...")
    hist_fused_fixed, (y_te_fused_fx, y_pred_fused_fx, cm_fused_fx) = train_keras_classifier_get_history(fused_fixed_all, y_labels, "Fused-Fixed", epochs=30)

    # ---------------------- EXACTLY 4 OVERALL CONFUSION MATRICES ----------------------
    plot_confusion(cm_hand, "Overall Confusion Matrix - Handcrafted (held-out)")
    plot_confusion(cm_cnn, "Overall Confusion Matrix - CNN (held-out)")
    plot_confusion(cm_fused_ad, "Overall Confusion Matrix - GStageNet-Adaptive (held-out)")
    plot_confusion(cm_fused_fx, "Overall Confusion Matrix - GStageNet-Fixed (held-out)")

    # ---------------------- Separate train/val accuracy & loss plots for each ablation case (4) ----------------------
    def plot_history_separate(history, title_prefix):
        epochs = len(history.history['loss'])
        fig, axes = plt.subplots(1,2, figsize=(12,4))
        axes[0].plot(range(1, epochs+1), history.history['accuracy'], label='train')
        axes[0].plot(range(1, epochs+1), history.history['val_accuracy'], linestyle='--', label='val')
        axes[0].set_title(f"{title_prefix} - Accuracy")
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy'); axes[0].legend(); axes[0].grid(True)

        axes[1].plot(range(1, epochs+1), history.history['loss'], label='train')
        axes[1].plot(range(1, epochs+1), history.history['val_loss'], linestyle='--', label='val')
        axes[1].set_title(f"{title_prefix} - Loss")
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss'); axes[1].legend(); axes[1].grid(True)
        plt.show()

    plot_history_separate(hist_handcrafted, "Handcrafted (MLP)")
    plot_history_separate(hist_cnn, "CNN embeddings (MLP)")
    plot_history_separate(hist_fused_adaptive, "Fused-Adaptive (MLP)")
    plot_history_separate(hist_fused_fixed, "Fused-Fixed (MLP)")

    # ---------------------- t-SNE VISUALIZATION (4) ----------------------
    print("\nComputing t-SNE projections for visualization (this may take a moment)...")
    tsne = TSNE(n_components=2, random_state=42)
    tsne_hand = tsne.fit_transform(feats_scaled)
    tsne_cnn = tsne.fit_transform(embeddings)
    tsne_fused_ad = tsne.fit_transform(fused_adaptive_all)
    tsne_fused_fx = tsne.fit_transform(fused_fixed_all)

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    for proj, title, ax in zip([tsne_hand, tsne_cnn, tsne_fused_ad, tsne_fused_fx],
                               ['t-SNE: Handcrafted', 't-SNE: CNN embeddings', 't-SNE: Fused-Adaptive', 't-SNE: Fused-Fixed'], axes):
        for cls in range(N_CLASSES):
            ax.scatter(proj[y_labels == cls, 0], proj[y_labels == cls, 1], s=10, label=CLASS_NAMES[cls])
        ax.set_title(title)
        ax.legend(markerscale=2)
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.show()

    # ---------------------- RAW FEATURE CORRELATION PLOTS (for all 4 representations) ----------------------
    print("\nPlotting raw feature correlation matrices (may be large for high-dim embeddings)...")
    plot_raw_correlations(feats_scaled, "Raw correlations: Handcrafted features", max_vars_to_plot=50)
    plot_raw_correlations(embeddings, "Raw correlations: CNN embeddings", max_vars_to_plot=100)
    plot_raw_correlations(fused_adaptive_all, "Raw correlations: Fused-Adaptive features", max_vars_to_plot=100)
    plot_raw_correlations(fused_fixed_all, "Raw correlations: Fused-Fixed features", max_vars_to_plot=100)

    # ---------------------- Print summary table ----------------------
    df = pd.DataFrame(summary, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score", "AUC", "Model Size (MB)"])
    print("\n🔎 Overall Performance Summary:")
    print(df)

    print("\nDone.")
