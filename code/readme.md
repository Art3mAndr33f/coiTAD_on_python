# coiTAD — Circle of Influence TAD Caller

Python-реализация алгоритма **coiTAD** для идентификации топологически ассоциированных доменов (TAD)
из Hi-C контактных матриц. Поддерживает кластеризацию **HDBSCAN** и **OPTICS** с биологической
валидацией на основе ChIP-Seq данных (ENCODE hESC).

> Оригинальный алгоритм: [OluwadareLab/coiTAD](https://github.com/OluwadareLab/coiTAD/tree/main)  
> Статья: [coiTAD: Circle of Influence-Based TAD Detection (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11507547/)

---

## Установка

```bash
# Рекомендуется: чистая conda-среда
conda create -n coitad python=3.11 -y
conda activate coitad
conda install -c conda-forge numpy scipy pandas matplotlib seaborn h5py -y
pip install cooler hdbscan scikit-learn requests
```

<details>
<summary>Альтернатива: pip + venv</summary>

```bash
python -m venv coitad_env
coitad_env\Scripts\activate        # Windows
# source coitad_env/bin/activate   # Linux/Mac

pip install numpy==1.26.4
pip install h5py cooler hdbscan scikit-learn scipy pandas matplotlib seaborn requests
```

> **Важно:** `numpy < 2.0` для совместимости с `h5py`/`cooler` на Windows.
</details>

---

## Структура проекта

```
code/
│
├── run.py                   # CLI — точка входа для одиночного запуска
├── run_batch.py             # Пакетный запуск: N хромосом × M разрешений × 2 метода
├── visualize_batch.py       # Визуализация результатов батча (без перезапуска)
│
├── pipeline.py              # Пайплайны: run_coitad / run_comparison / run_full_analysis
├── coitad.py                # Алгоритм coiTAD (базовый класс + HDBSCAN + OPTICS)
├── feature_generation.py    # Генерация circle-of-influence признаков
├── extract_tad.py           # Извлечение TAD из кластеров
├── quality_check.py         # Оценка качества TAD (intra/inter contact frequency)
├── mcool_converter.py       # Конвертация .mcool → текстовая матрица
├── comparison.py            # Сравнение методов (MoC, ARI, NMI, Precision/Recall/F1)
├── validation.py            # Биологическая валидация (ChIP-Seq) + загрузка ENCODE
├── visualization.py         # Визуализация контактных карт и TAD
├── utils.py                 # Утилиты (BED-конвертер и пр.)
│
└── README.md
```

---

## Быстрый старт

### Одиночный запуск (CLI)

```bash
# HDBSCAN (по умолчанию)
python run.py single 4DNFI52OLNJ4.mcool --chr chr19 --res 50000

# OPTICS
python run.py single 4DNFI52OLNJ4.mcool --chr chr19 --res 50000 --method OPTICS

# Сравнение HDBSCAN vs OPTICS
python run.py compare 4DNFI52OLNJ4.mcool --chr chr19 --res 50000

# Полный анализ с биологической валидацией
python run.py full 4DNFI52OLNJ4.mcool --chr chr19 --res 50000 --genome hg19
```

### Одиночный запуск (Python API)

```python
from pipeline import run_coitad, run_comparison, run_full_analysis

# Один метод
run_coitad("4DNFI52OLNJ4.mcool", chromosome="chr19", resolution=50000)

# Сравнение двух методов
run_comparison("4DNFI52OLNJ4.mcool", chromosome="chr19", resolution=50000)

# Полный анализ + ChIP-Seq валидация
run_full_analysis("4DNFI52OLNJ4.mcool", chromosome="chr19", resolution=50000, genome="hg19")
```

---

## Пакетный запуск

Скрипт `run_batch.py` запускает все комбинации (хромосома × разрешение × метод)
и агрегирует результаты в единые таблицы.

### Настройка

Отредактируйте конфигурацию в начале `run_batch.py`:

```python
MCOOL_FILE   = "4DNFI52OLNJ4.mcool"
GENOME       = "hg19"
ROOT_OUTPUT  = Path("batch_results")

CHROMOSOMES  = ["chr19", "chr17", "chr22"]   # chr1 опционально (долго)
RESOLUTIONS  = [25000, 50000, 100000]
METHODS      = ["HDBSCAN", "OPTICS"]
```

### Запуск

```bash
python run_batch.py
```

### Что происходит

1. **Загрузка ChIP-Seq** — один раз скачиваются CTCF, H3K4me1, H3K4me3, H3K27ac из ENCODE
2. **Для каждой комбинации** (chr × res × method):
   - конвертация `.mcool` → текстовая матрица (пропускается если файл уже есть)
   - генерация circle-of-influence признаков
   - кластеризация (HDBSCAN или OPTICS)
   - извлечение TAD + оценка качества
   - расчёт ChIP-Seq обогащения на границах
3. **Попарное сравнение** HDBSCAN vs OPTICS для каждого (chr, res)
4. **Агрегация** — сводные таблицы и текстовый отчёт

---

## Визуализация результатов

Скрипт `visualize_batch.py` генерирует все графики из **уже сохранённых** результатов.
Ничего не пересчитывается — только чтение `.hic` матриц и `TAD_BinID.txt` файлов.

```bash
python visualize_batch.py
```

### Генерируемые визуализации

#### Для каждой комбинации (chr × res)

| Визуализация | Файл | Описание |
|---|---|---|
| Side-by-side | `chr19_50kb_side_by_side.png` | Две контактные карты рядом: HDBSCAN и OPTICS |
| Overlay | `chr19_50kb_overlay.png` | Одна карта, оба метода наложены (синий / красный) |
| OPTICS-only | `chr19_50kb_optics_only.png` | Полная карта: красным — TAD уникальные для OPTICS, серым — общие, синим — уникальные для HDBSCAN |
| OPTICS-only zoom | `chr19_50kb_optics_only_zoom.png` | Увеличенные панели каждого уникального OPTICS TAD с ближайшими HDBSCAN TAD |

#### Агрегированные (все условия)

| Визуализация | Файл | Описание |
|---|---|---|
| TAD counts | `summary_tad_counts.png` | Grouped bar: кол-во TAD по всем условиям |
| Size distributions | `size_distributions_50kb.png` | Гистограммы размеров TAD по хромосомам (один файл на разрешение) |
| Enrichment | `summary_enrichment.png` | ChIP-Seq обогащение (усреднённое по хромосомам и разрешениям) |
| Concordance | `summary_concordance.png` | MoC / ARI / NMI heatmap по всем условиям |

---

## Выходная структура

### Одиночный запуск (`run.py` / `pipeline.py`)

```
coitad_output/
├── data/
│   └── chr19_50kb.hic                   # Контактная матрица (текст)
├── features/
│   └── feature_radius_*.txt             # Circle-of-influence признаки
├── results/
│   ├── TADs/
│   │   ├── HDBSCAN_{r}_TAD_BinID.txt   # Границы TAD (bin ID)
│   │   └── HDBSCAN_{r}_domain.txt      # Границы TAD (геномные координаты)
│   ├── Quality/
│   │   └── Readme.txt                   # Рекомендуемый радиус
│   └── visualizations/                  # Графики (при visualize=True)
```

### Полный анализ (`run.py full`)

```
full_comparison/
├── data/
├── results_hdbscan/
├── results_optics/
├── comparison/
│   ├── tad_count_comparison.png
│   ├── tad_size_comparison.png
│   ├── moc_heatmap.png
│   └── comparison_report.txt
├── biological_validation/
│   ├── validation_results.csv
│   ├── enrichment_profiles.png
│   ├── peaks_per_bin_comparison.png
│   └── validation_report.txt
└── chipseq_data/
```

### Пакетный запуск (`run_batch.py` + `visualize_batch.py`)

```
batch_results/
├── chipseq_data/                           # ENCODE ChIP-Seq (общие)
│   ├── CTCF_hg19.bed
│   ├── H3K4me1_hg19.bed
│   ├── H3K4me3_hg19.bed
│   └── H3K27ac_hg19.bed
│
├── chr19_25kb/                             # По папке на каждое (chr, res)
│   ├── data/
│   │   └── chr19_25kb.hic
│   ├── features_HDBSCAN/
│   ├── features_OPTICS/
│   ├── results_HDBSCAN/
│   │   ├── TADs/
│   │   └── Quality/
│   ├── results_OPTICS/
│   │   ├── TADs/
│   │   └── Quality/
│   └── validation/
│
├── chr19_50kb/
│   └── ...
├── chr19_100kb/
│   └── ...
├── chr17_50kb/
│   └── ...
├── chr22_50kb/
│   └── ...
│
├── all_results.csv                         # Сводная таблица всех запусков
├── all_comparisons.csv                     # HDBSCAN vs OPTICS на каждом условии
├── aggregate_report.txt                    # Итоговый текстовый отчёт
│
└── visualizations/                         # Генерируется visualize_batch.py
    ├── chr19_25kb/
    │   ├── chr19_25kb_side_by_side.png
    │   ├── chr19_25kb_overlay.png
    │   ├── chr19_25kb_optics_only.png
    │   └── chr19_25kb_optics_only_zoom.png
    ├── chr19_50kb/
    │   └── ...
    ├── chr17_50kb/
    │   └── ...
    ├── summary_tad_counts.png
    ├── summary_enrichment.png
    ├── summary_concordance.png
    ├── size_distributions_25kb.png
    ├── size_distributions_50kb.png
    └── size_distributions_100kb.png
```

---

## Как устроен алгоритм

```
.mcool ──► mcool_converter ──► матрица контактов
                                      │
                                      ▼
                              feature_generation
                           (circle of influence)
                                      │
                            ┌─────────┴─────────┐
                            ▼                   ▼
                         HDBSCAN             OPTICS
                            │                   │
                            └─────────┬─────────┘
                                      ▼
                                 extract_tad
                              (границы TAD)
                                      │
                                      ▼
                               quality_check
                          (intra vs inter IF)
                                      │
                        ┌─────────────┼─────────────┐
                        ▼             ▼             ▼
                  comparison     validation    visualization
                 (MoC, ARI,    (ChIP-Seq     (контактные
                  NMI, P/R)    enrichment)      карты)
```

1. **Feature Generation** — для каждой точки на диагонали матрицы строится вектор признаков
   по 8 направлениям (circle of influence) с радиусом от `min_radius` до `max_radius`
2. **Clustering** — признаки кластеризуются для каждого радиуса; смежные бины
   с одинаковой меткой объединяются в TAD
3. **Quality Check** — для каждого радиуса вычисляется разность средних intra-TAD
   и inter-TAD контактных частот; лучший радиус — максимальная разность
4. **Comparison** — MoC (совпадение границ), ARI и NMI (кластерные метрики),
   Precision/Recall/F1 (граничная точность)
5. **Validation** — подсчёт пиков ChIP-Seq (CTCF, H3K4me1, H3K4me3, H3K27ac)
   в позициях TAD-границ; больше пиков = границы биологически обоснованы

---

## Входные данные

| Формат | Описание | Источник |
|---|---|---|
| `.mcool` | Мульти-разрешенные Hi-C контактные матрицы | [4DN Data Portal](https://data.4dnucleome.org/) |
| Текстовая матрица | Квадратная матрица контактов (пробел/таб) | Любой Hi-C pipeline |
| ChIP-Seq BED | Пиковые файлы для валидации (опционально) | [ENCODE](https://www.encodeproject.org/) |

> Тестовый датасет: **4DNFI52OLNJ4.mcool** — H1-hESC, DpnII, hg19-совместимый

---

## Ключевые параметры

| Параметр | По умолчанию | Описание |
|---|---|---|
| `resolution` | 50000 | Разрешение Hi-C (bp) |
| `max_tad_size` | 800000 | Максимальный размер TAD (bp) |
| `min_samples` (OPTICS) | 5 | Мин. точек в окрестности |
| `xi` (OPTICS) | 0.05 | Крутизна reachability-кривой |
| `min_cluster_size` (OPTICS) | 0.05 | Мин. размер кластера (доля) |
| `tolerance` (сравнение) | 2 | Допуск на совпадение границ (в бинах) |

---

## Типичный рабочий процесс

```bash
# 1. Пакетный запуск на нескольких хромосомах и разрешениях
python run_batch.py

# 2. Визуализация (без перезапуска вычислений)
python visualize_batch.py

# 3. Результаты:
#    batch_results/all_results.csv          — сводная таблица
#    batch_results/aggregate_report.txt     — текстовый отчёт
#    batch_results/visualizations/          — все графики
```