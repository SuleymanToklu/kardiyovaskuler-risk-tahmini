"""
fuzzification.py
----------------
Makale reprodüksiyonu projesi — Adım 3: Bulanıklaştırma (Fuzzification).

other_factors kuralları:
  smoke=0, alco=0, active=1  → 0  (sağlıklı)
  smoke=1, alco=1, active=0  → 1  (sağlıksız)
  diğer tüm kombinasyonlar   → 2  (belirsiz/karışık)

cardio kuralları (sıfırdan üretilir, orijinal sütun kullanılmaz):
  Kadın (gender=1) yaş eşiği: 55
  Erkek (gender=2) yaş eşiği: 45
  age < eşik  VE other_factors == 0  → 0  (risksiz)
  age >= eşik VE other_factors == 1  → 1  (yüksek risk)
  diğer tüm kombinasyonlar            → 2  (orta/belirsiz risk)

Kullanım:
  python fuzzification.py

Bu modül şu adımları gerçekleştirir:
  1. `heart_disease.db` veritabanından `cleaned_data` tablosunu okur.
  2. `smoke`, `alco`, `active` sütunlarından `other_factors` üretir.
  3. `cardio` hedef değişkenini bulanık üyelik değerine (0 / 0.5 / 1) dönüştürür.
  4. Sonucu `processed_data` tablosu olarak aynı veritabanına kaydeder.

──────────────────────────────────────────────────────────────────
  other_factors kuralları (bulanık OR mantığı)
  ─────────────────────────────────────────────
  Sağlıklı referans değerleri:
    smoke  = 0  (içmiyor)
    alco   = 0  (içmiyor)
    active = 1  (fiziksel olarak aktif)

  Sağlıksız referans değerleri:
    smoke  = 1  (içiyor)
    alco   = 1  (içiyor)
    active = 0  (aktif değil)

  Sağlıklı bayrak (healthy_flag) = smoke==0 ve alco==0 ve active==1
  Sağlıksız bayrak (unhealthy_flag) = smoke==1 ve alco==1 ve active==0

  └─ Tümü sağlıklı  (healthy_flag=True)  → other_factors = 0.0
  └─ Tümü sağlıksız (unhealthy_flag=True) → other_factors = 1.0
  └─ Diğer (uyumsuz/karışık)             → other_factors = 0.5

──────────────────────────────────────────────────────────────────
  cardio yeniden kodlama kuralları
  ─────────────────────────────────
  Kaynak: kardiyovasküler bulanık mantık literatürü
  (Örn. Zadeh tabanlı hastalık risk sınıflandırması)

  ┌──────────────┬─────────────────────────────────────┬─────────────────┐
  │ cardio_orig  │ Ek koşul                            │ cardio_fuzzy    │
  ├──────────────┼─────────────────────────────────────┼─────────────────┤
  │      0       │ —                                   │     0.0         │
  │      1       │ age ≥ 55 VE other_factors ≥ 0.5    │     1.0  (yüksek risk) │
  │      1       │ Diğer tüm durumlar                  │     0.5  (orta risk)   │
  └──────────────┴─────────────────────────────────────┴─────────────────┘

  Gerekçe:
  - 55 yaş üstü bireylerde kardiyovasküler hastalık prevalansı belirgin
    biçimde artar (AHA/ACC 2019 kılavuzu eşik yaşı).
  - Aynı zamanda sağlıksız yaşam alışkanlığı (other_factors ≥ 0.5) mevcutsa
    risk "kesin yüksek (1.0)" kategorisine girer.
  - Bu koşulları sağlamayan cardio=1 gözlemler "olası / orta risk (0.5)"
    olarak kodlanır; model bu örnekleri kısmi üyelikle değerlendirir.

Kullanım:
  python fuzzification.py
"""

import sqlite3
import logging
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Yapılandırma
# ---------------------------------------------------------------------------

BASE_DIR  = Path(__file__).parent
DB_PATH   = BASE_DIR / "heart_disease.db"
SRC_TABLE = "cleaned_data"
DST_TABLE = "processed_data"

# Cinsiyete göre yaş eşikleri
AGE_THRESHOLD_FEMALE: int = 55  # gender == 1
AGE_THRESHOLD_MALE:   int = 45  # gender == 2

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
# Adım fonksiyonları
# ---------------------------------------------------------------------------

def load_cleaned_data(db_path: Path, table: str) -> pd.DataFrame:
    """SQLite veritabanından temizlenmiş veriyi okur.

    Parameters
    ----------
    db_path : Path
        SQLite veritabanı dosyasının yolu.
    table : str
        Okunacak tablo adı.

    Returns
    -------
    pd.DataFrame
        Temizlenmiş veri içeren DataFrame.

    Raises
    ------
    FileNotFoundError
        Veritabanı dosyası bulunamazsa fırlatılır.
    """
    if not db_path.is_file():
        raise FileNotFoundError(f"Veritabanı bulunamadı: {db_path}")

    logger.info("Veritabanından okunuyor: %s → tablo: '%s'", db_path, table)
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            f"SELECT * FROM {table}", conn, index_col="id"
        )

    logger.info(
        "Veri yüklendi → %d satır, %d sütun. Sütunlar: %s",
        len(df),
        len(df.columns),
        df.columns.tolist(),
    )
    return df


def compute_other_factors(df: pd.DataFrame) -> pd.DataFrame:
    """'smoke', 'alco', 'active' sütunlarından 'other_factors' türetir.

    Kural tablosu:
      smoke=0, alco=0, active=1  → 0  (sağlıklı)
      smoke=1, alco=1, active=0  → 1  (sağlıksız)
      diğer tüm kombinasyonlar   → 2  (belirsiz/karışık)

    Parameters
    ----------
    df : pd.DataFrame
        'smoke', 'alco', 'active' sütunları olan DataFrame.

    Returns
    -------
    pd.DataFrame
        'other_factors' (int: 0/1/2) sütunu eklenmiş DataFrame.
    """
    df = df.copy()

    healthy   = (df["smoke"] == 0) & (df["alco"] == 0) & (df["active"] == 1)
    unhealthy = (df["smoke"] == 1) & (df["alco"] == 1) & (df["active"] == 0)

    df["other_factors"] = 2                  # varsayılan: belirsiz
    df.loc[healthy,   "other_factors"] = 0   # tümü sağlıklı
    df.loc[unhealthy, "other_factors"] = 1   # tümü sağlıksız
    df["other_factors"] = df["other_factors"].astype(int)

    counts = df["other_factors"].value_counts().sort_index()
    logger.info(
        "other_factors dağılımı:\n%s",
        counts.rename({0: "0 (sağlıklı)", 1: "1 (sağlıksız)", 2: "2 (karışık)"}).to_string(),
    )
    return df


def build_cardio_target(df: pd.DataFrame) -> pd.DataFrame:
    """'cardio' hedef sütununu cinsiyete özgü yaş eşikleriyle sıfırdan üretir.

    Orijinal 'cardio' sütunu yok sayılır; kural tablosu:
      Kadın (gender=1): yaş eşiği = AGE_THRESHOLD_FEMALE (55)
      Erkek (gender=2): yaş eşiği = AGE_THRESHOLD_MALE   (45)

      age < eşik  VE other_factors == 0  → 0  (risksiz)
      age >= eşik VE other_factors == 1  → 1  (yüksek risk)
      diğer tüm kombinasyonlar            → 2  (orta/belirsiz risk)

    Parameters
    ----------
    df : pd.DataFrame
        'gender', 'age', 'other_factors' sütunları olan DataFrame.

    Returns
    -------
    pd.DataFrame
        'cardio' sütunu 0/1/2 tamsayılarla güncellenmiş DataFrame.
    """
    df = df.copy()

    # Cinsiyete göre yaş eşiği sütunu
    age_thresh = df["gender"].map({
        1: AGE_THRESHOLD_FEMALE,
        2: AGE_THRESHOLD_MALE,
    })

    # Varsayılan: orta/belirsiz risk
    df["cardio"] = 2

    # Risksiz: yaş eşiğinin ALTINDA VE sağlıklı alışkanlık
    low_risk  = (df["age"] < age_thresh) & (df["other_factors"] == 0)
    # Yüksek risk: yaş eşiğinin ÜSTÜNDE VE sağlıksız alışkanlık
    high_risk = (df["age"] >= age_thresh) & (df["other_factors"] == 1)

    df.loc[low_risk,  "cardio"] = 0
    df.loc[high_risk, "cardio"] = 1
    df["cardio"] = df["cardio"].astype(int)

    counts = df["cardio"].value_counts().sort_index()
    logger.info(
        "cardio (yeni etiket) dağılımı:\n%s",
        counts.rename({0: "0 (risksiz)", 1: "1 (yüksek risk)", 2: "2 (orta risk)"}).to_string(),
    )
    pct = counts / len(df) * 100
    logger.info(
        "Yüzde → 0: %.1f%%  1: %.1f%%  2: %.1f%%",
        pct.get(0, 0), pct.get(1, 0), pct.get(2, 0),
    )
    return df


def save_processed_data(
    df: pd.DataFrame,
    db_path: Path,
    table: str,
    if_exists: str = "replace",
) -> None:
    """İşlenmiş veriyi SQLite veritabanına yazar.

    Parameters
    ----------
    df : pd.DataFrame
        Kaydedilecek DataFrame.
    db_path : Path
        Hedef SQLite veritabanı dosyasının yolu.
    table : str
        Hedef tablo adı.
    if_exists : str, optional
        Tablo varsa: 'replace' (varsayılan) | 'append' | 'fail'.
    """
    logger.info("İşlenmiş veri '%s' tablosuna yazılıyor…", table)
    with sqlite3.connect(db_path) as conn:
        df.to_sql(name=table, con=conn, if_exists=if_exists, index=True)
        row_count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    logger.info(
        "Tablo '%s' başarıyla kaydedildi → %d kayıt",
        table,
        row_count,
    )


def verify_processed_data(db_path: Path, table: str) -> None:
    """Kaydedilen `processed_data` tablosunu doğrular ve özet loglar.

    Parameters
    ----------
    db_path : Path
        SQLite veritabanı dosyasının yolu.
    table : str
        Doğrulanacak tablo adı.
    """
    with sqlite3.connect(db_path) as conn:
        schema  = pd.read_sql_query(f"PRAGMA table_info({table})", conn)
        preview = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 8", conn)
        dist    = pd.read_sql_query(
            f"SELECT cardio, other_factors, COUNT(*) AS count "
            f"FROM {table} "
            f"GROUP BY cardio, other_factors "
            f"ORDER BY cardio, other_factors",
            conn,
        )

    logger.info(
        "Tablo şeması:\n%s",
        schema[["name", "type"]].to_string(index=False),
    )
    logger.info("İlk 8 satır:\n%s", preview.to_string(index=False))
    logger.info(
        "cardio × other_factors çapraz dağılımı:\n%s",
        dist.to_string(index=False),
    )


# ---------------------------------------------------------------------------
# Ana iş akışı
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    """Bulanıklaştırma sürecini baştan sona çalıştırır."""
    logger.info("=" * 60)
    logger.info("Bulanıklaştırma (Fuzzification) başlıyor.")
    logger.info("Kadın yaş eşiği=%d, Erkek yaş eşiği=%d",
                AGE_THRESHOLD_FEMALE, AGE_THRESHOLD_MALE)
    logger.info("=" * 60)

    # 1. Temizlenmiş veriyi yükle
    df = load_cleaned_data(DB_PATH, SRC_TABLE)

    # 2. other_factors sütununu oluştur (0/1/2 tamsayı)
    logger.info("─" * 40)
    logger.info("Adım 1 — other_factors hesaplanıyor…")
    df = compute_other_factors(df)

    # 3. cardio sütununu sıfırdan üret (cinsiyet bazlı eşik, 0/1/2 tamsayı)
    logger.info("─" * 40)
    logger.info("Adım 2 — cardio hedefi sıfırdan üretiliyor…")
    df = build_cardio_target(df)

    # 4. processed_data olarak kaydet
    logger.info("─" * 40)
    save_processed_data(df, DB_PATH, DST_TABLE, if_exists="replace")

    # 5. Doğrula
    verify_processed_data(DB_PATH, DST_TABLE)

    logger.info("=" * 60)
    logger.info(
        "Fuzzification tamamlandı. Tablo: '%s' @ %s",
        DST_TABLE,
        DB_PATH,
    )
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Giriş noktası
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline()
