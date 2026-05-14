"""
model_training.py
-----------------
Makale reprodüksiyonu projesi — Adım 5: Model Eğitimi ve Karşılaştırma.

Bu modül şu adımları gerçekleştirir:
  1. `processed_data` tablosunu SQLite'dan okur.
  2. İki farklı senaryo için özellik/hedef setlerini hazırlar:
       - Normal : smoke / alco / active (binary) + binary cardio (0/1)
       - Fuzzy  : other_factors + üç sınıflı cardio (0 / 1 / 2)
  3. 7 sınıflandırıcıyı (GNB, SVM, AdaBoost, DT, KNN, RF, GB) her iki
     senaryo için eğitir ve şu metriklerle değerlendirir:
       - Accuracy, Macro-F1, Macro-Precision
       - ROC-AUC  (Normal: binary; Fuzzy: OvR macro)
       - 5-Katlı Çapraz Doğrulama Accuracy (CV_MAX_SAMPLES üzerinden)
       - Hesaplama Süresi (train + predict)
  4. Sonuçları üç tabloya kaydeder:
       - model_results      : temel metrikler
       - confusion_matrices : her model için karmaşıklık matrisi (JSON)
       - roc_curves         : Normal senaryosu FPR/TPR eğrileri (JSON)

Özellik setleri
  Normal  → age, gender, ap_hi, ap_lo, cholesterol, gluc,
             smoke, alco, active, bmi
  Fuzzy   → gender, age, other_factors

Hedef değişkenler
  Normal  → cardio_binary : processed_data["cardio"] > 0 → 1, aksi → 0
  Fuzzy   → cardio (0/1/2 tamsayı)

Kullanım:
  python model_training.py
"""

import json
import sys
import time
import sqlite3
import logging
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# ---------------------------------------------------------------------------
# Yapılandırma
# ---------------------------------------------------------------------------

BASE_DIR   = Path(__file__).parent
DB_PATH    = BASE_DIR / "heart_disease.db"
SRC_TABLE  = "processed_data"
RES_TABLE  = "model_results"

RANDOM_STATE   = 42
TEST_SIZE      = 0.20
CV_N_SPLITS    = 5
CV_MAX_SAMPLES = 3_000    # CV için kullanılacak maksimum örnek sayısı

NORMAL_FEATURES: list[str] = [
    "age", "gender", "ap_hi", "ap_lo",
    "cholesterol", "gluc",
    "smoke", "alco", "active",
    "bmi",
]

FUZZY_FEATURES: list[str] = [
    "gender",
    "age",
    "other_factors",
]

# ---------------------------------------------------------------------------
# Loglama
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sınıflandırıcı tanımları
# ---------------------------------------------------------------------------

def build_classifiers() -> dict:
    """7 sınıflandırıcıyı isim → nesne sözlüğü olarak döner.

    SVM için LinearSVC + CalibratedClassifierCV kullanılır: RBF kernel'e kıyasla
    büyük veri setlerinde çok daha hızlı eğitilir ve predict_proba desteği sağlar.
    """
    return {
        "GNB"     : GaussianNB(),
        "SVM"     : CalibratedClassifierCV(
                        LinearSVC(C=1.0, dual="auto", max_iter=5000, random_state=RANDOM_STATE),
                        cv=3,
                    ),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "DT"      : DecisionTreeClassifier(random_state=RANDOM_STATE),
        "KNN"     : KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "RF"      : RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "GB"      : GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
    }


# ---------------------------------------------------------------------------
# Veri yükleme
# ---------------------------------------------------------------------------

def load_processed_data(db_path: Path, table: str) -> pd.DataFrame:
    """SQLite veritabanından işlenmiş veriyi okur."""
    if not db_path.is_file():
        raise FileNotFoundError(f"Veritabanı bulunamadı: {db_path}")

    logger.info("Veritabanından okunuyor: %s → tablo: '%s'", db_path, table)
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn, index_col="id")

    logger.info("Veri yüklendi → %d satır, %d sütun", len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Veri hazırlama
# ---------------------------------------------------------------------------

def prepare_scenarios(df: pd.DataFrame) -> dict:
    """Normal ve Fuzzy senaryolar için X/y çiftlerini hazırlar."""
    y_binary = (df["cardio"] > 0).astype(int)
    y_fuzzy  = df["cardio"].astype(int)
    X_fuzzy  = df[FUZZY_FEATURES].copy().astype(int)

    scenarios = {
        "Normal": {
            "X": df[NORMAL_FEATURES].copy(),
            "y": y_binary,
            "label": "Binary cardio (0/1)",
            "is_binary": True,
        },
        "Fuzzy": {
            "X": X_fuzzy,
            "y": y_fuzzy,
            "label": "Multiclass cardio (0/1/2) — gender+age+other_factors",
            "is_binary": False,
        },
    }

    for name, s in scenarios.items():
        logger.info(
            "Senaryo '%s': X şekli=%s | y dağılımı:\n%s",
            name, s["X"].shape,
            s["y"].value_counts().sort_index().to_string(),
        )

    return scenarios


# ---------------------------------------------------------------------------
# Eğitim ve değerlendirme
# ---------------------------------------------------------------------------

def _compute_cv(
    clf,
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    clf_name: str,
    scenario_name: str,
) -> tuple[float | None, float | None]:
    """Stratified k-katlı çapraz doğrulama accuracy'sini hesaplar.

    Büyük veri setlerinde hızı dengelemek için CV_MAX_SAMPLES kadar örnek kullanır.
    """
    n = min(len(X), CV_MAX_SAMPLES)
    if len(X) > CV_MAX_SAMPLES:
        logger.info(
            "  CV için örnekleme: %d / %d satır kullanılıyor.", n, len(X)
        )
        sample_idx = X.sample(n=n, random_state=RANDOM_STATE).index
        X_cv, y_cv = X.loc[sample_idx], y.loc[sample_idx]
    else:
        X_cv, y_cv = X, y

    try:
        scores = cross_val_score(clone(clf), X_cv, y_cv, cv=cv, scoring="accuracy")
        return round(float(scores.mean()), 4), round(float(scores.std()), 4)
    except Exception as exc:
        logger.warning("CV başarısız (%s/%s): %s", scenario_name, clf_name, exc)
        return None, None


def train_and_evaluate(
    classifiers: dict,
    scenarios: dict,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tüm sınıflandırıcıları her iki senaryo için eğitip değerlendirir.

    Returns
    -------
    (results_df, cm_df, roc_df)
      results_df : temel metrikler tablosu
      cm_df      : karmaşıklık matrisleri (JSON sütunlu)
      roc_df     : ROC eğrisi verileri (Normal senaryo, JSON sütunlu)
    """
    records:     list[dict] = []
    cm_records:  list[dict] = []
    roc_records: list[dict] = []

    cv = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=random_state)

    for scenario_name, scenario in scenarios.items():
        X = scenario["X"]
        y = scenario["y"]
        is_binary = scenario["is_binary"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        logger.info(
            "─── Senaryo: %-8s  | train=%d  test=%d ───",
            scenario_name, len(X_train), len(X_test),
        )

        for clf_name, clf in classifiers.items():
            logger.info("  İşleniyor: %s / %s …", scenario_name, clf_name)

            # 1. Çapraz doğrulama
            cv_mean, cv_std = _compute_cv(clf, X, y, cv, clf_name, scenario_name)

            # 2. Train / test
            t_start = time.perf_counter()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            elapsed = time.perf_counter() - t_start

            acc  = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
            f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)

            # 3. ROC-AUC
            roc_auc = None
            if hasattr(clf, "predict_proba"):
                try:
                    if is_binary:
                        y_prob  = clf.predict_proba(X_test)[:, 1]
                        roc_auc = round(roc_auc_score(y_test, y_prob), 4)
                        fpr, tpr, _ = roc_curve(y_test, y_prob)
                        roc_records.append({
                            "Scenario"  : scenario_name,
                            "Classifier": clf_name,
                            "FPR"       : json.dumps([round(v, 4) for v in fpr.tolist()]),
                            "TPR"       : json.dumps([round(v, 4) for v in tpr.tolist()]),
                            "AUC"       : roc_auc,
                        })
                    else:
                        y_prob  = clf.predict_proba(X_test)
                        roc_auc = round(
                            roc_auc_score(
                                y_test, y_prob,
                                multi_class="ovr", average="macro",
                            ), 4,
                        )
                except Exception as exc:
                    logger.warning(
                        "ROC-AUC hesaplanamadı (%s/%s): %s",
                        scenario_name, clf_name, exc,
                    )

            # 4. Karmaşıklık matrisi
            labels = sorted(y_test.unique().tolist())
            cm     = confusion_matrix(y_test, y_pred, labels=labels)
            cm_records.append({
                "Scenario"  : scenario_name,
                "Classifier": clf_name,
                "Matrix"    : json.dumps(cm.tolist()),
                "Labels"    : json.dumps(labels),
            })

            records.append({
                "Scenario"          : scenario_name,
                "Classifier"        : clf_name,
                "Accuracy"          : round(acc,  4),
                "Macro_Precision"   : round(prec, 4),
                "Macro_F1"          : round(f1,   4),
                "ROC_AUC"           : roc_auc,
                "CV_Acc_Mean"       : cv_mean,
                "CV_Acc_Std"        : cv_std,
                "Computation_Time_s": round(elapsed, 4),
            })

            logger.info(
                "    ✓ Acc=%.4f | F1=%.4f | AUC=%s | CV=%.4f±%.4f | %.2fs",
                acc, f1,
                f"{roc_auc:.4f}" if roc_auc is not None else "N/A",
                cv_mean or 0.0, cv_std or 0.0,
                elapsed,
            )

    results_df = pd.DataFrame(records)
    cm_df      = pd.DataFrame(cm_records)
    roc_df     = (
        pd.DataFrame(roc_records)
        if roc_records
        else pd.DataFrame(columns=["Scenario", "Classifier", "FPR", "TPR", "AUC"])
    )
    return results_df, cm_df, roc_df


# ---------------------------------------------------------------------------
# Kaydetme ve raporlama
# ---------------------------------------------------------------------------

def save_results(
    results_df: pd.DataFrame,
    cm_df: pd.DataFrame,
    roc_df: pd.DataFrame,
    db_path: Path,
    table: str,
) -> None:
    """Sonuç tablolarını SQLite veritabanına yazar."""
    with sqlite3.connect(db_path) as conn:
        results_df.to_sql(name=table,               con=conn, if_exists="replace", index=False)
        cm_df.to_sql(     name="confusion_matrices", con=conn, if_exists="replace", index=False)
        if not roc_df.empty:
            roc_df.to_sql(name="roc_curves",         con=conn, if_exists="replace", index=False)

    logger.info(
        "Kaydedildi: '%s' (%d satır), 'confusion_matrices' (%d satır), 'roc_curves' (%d satır)",
        table, len(results_df), len(cm_df), len(roc_df),
    )


def print_results(df: pd.DataFrame) -> None:
    """Sonuç tablosunu biçimlendirilmiş şekilde konsola basar."""
    sep = "=" * 90
    print(f"\n{sep}")
    print("  MODEL KARŞILAŞTIRMA SONUÇLARI")
    print(sep)

    for scenario, group in df.groupby("Scenario", sort=False):
        print(f"\n  Senaryo: {scenario}")
        print("-" * 80)
        display = group.drop(columns="Scenario").set_index("Classifier")
        display.columns = ["Accuracy", "Macro-Prec", "Macro-F1", "ROC-AUC",
                           "CV-Acc", "CV-Std", "Süre (s)"]
        print(display.to_string())

    print(f"\n{sep}")

    for scenario, group in df.groupby("Scenario", sort=False):
        best_acc  = group.loc[group["Accuracy"].idxmax()]
        best_f1   = group.loc[group["Macro_F1"].idxmax()]
        fastest   = group.loc[group["Computation_Time_s"].idxmin()]
        print(f"  [{scenario}] En yuksek Accuracy  -> {best_acc['Classifier']}  ({best_acc['Accuracy']:.4f})")
        print(f"  [{scenario}] En yuksek Macro-F1  -> {best_f1['Classifier']}  ({best_f1['Macro_F1']:.4f})")
        print(f"  [{scenario}] En hizli model      -> {fastest['Classifier']}  ({fastest['Computation_Time_s']:.4f}s)")

    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Ana iş akışı
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    """Model eğitimi ve karşılaştırma sürecini baştan sona çalıştırır."""
    logger.info("=" * 60)
    logger.info("Model Eğitimi ve Karşılaştırma başlıyor.")
    logger.info("7 sınıflandırıcı × 2 senaryo = 14 kombinasyon")
    logger.info("CV: %d katlı, maks. %d örnek", CV_N_SPLITS, CV_MAX_SAMPLES)
    logger.info("=" * 60)

    df = load_processed_data(DB_PATH, SRC_TABLE)

    logger.info("─" * 40)
    logger.info("Veri senaryoları hazırlanıyor…")
    scenarios = prepare_scenarios(df)

    classifiers = build_classifiers()
    logger.info("Sınıflandırıcılar: %s", list(classifiers.keys()))

    logger.info("─" * 40)
    results_df, cm_df, roc_df = train_and_evaluate(classifiers, scenarios)

    logger.info("─" * 40)
    save_results(results_df, cm_df, roc_df, DB_PATH, RES_TABLE)

    print_results(results_df)

    logger.info("=" * 60)
    logger.info("Pipeline tamamlandı. Sonuçlar '%s' @ %s", RES_TABLE, DB_PATH)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Giriş noktası
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline()
