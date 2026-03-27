# coiTAD — Circle of Influence TAD Caller

Python-реализация алгоритма **coiTAD** для идентификации топологически ассоциированных доменов (TAD)
из Hi-C контактных матриц. Поддерживает кластеризацию **HDBSCAN** и **OPTICS** с биологической
валидацией на основе ChIP-Seq данных (ENCODE).

> Оригинальный алгоритм: [OluwadareLab/coiTAD](https://github.com/OluwadareLab/coiTAD/tree/main)
> Статья: [coiTAD (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11507547/)

---

## Установка

```bash
conda create -n coitad python=3.11 -y
conda activate coitad
conda install -c conda-forge numpy scipy pandas matplotlib seaborn h5py -y
pip install cooler hdbscan scikit-learn requests
```

<details>
<summary>pip + venv (Windows)</summary>

```bash
python -m venv coitad_env
coitad_env\Scripts\activate
pip install numpy==1.26.4 h5py cooler hdbscan scikit-learn scipy pandas matplotlib seaborn requests
```
</details>

---

## Входные данные

Файлы `.mcool` размещаются в папке `data/`:

```
data/
├── 4DNFI52OLNJ4.mcool    # H1-hESC, in situ Hi-C, DpnII
├── 4DNFIHFX73VQ.mcool    # (второй датасет)
└── ...                    # любые другие .mcool
```

Скачать с [4DN Data Portal](https://data.4dnucleome.org/).

> **Важно:** ChIP-Seq валидация использует ENCODE hESC данные (hg19).
> Для корректности оба датасета должны быть от **той же клеточной линии**.

---

## Структура проекта

```
code/
├── run.py                   # CLI — одиночный запуск
├── run_batch.py             # Пакетный запуск (N датасетов × хромосомы × разрешения × методы)
├── visualize_batch.py       # Визуализация (без перезапуска)
├── tune_optics.py           # Тюнинг OPTICS: --mode simple | cv
│
├── pipeline.py              # run_coitad / run_comparison / run_full_analysis
├── coitad.py                # Алгоритм (базовый класс + HDBSCAN + OPTICS)
├── feature_generation.py    # Circle-of-influence признаки
├── extract_tad.py           # Извлечение TAD из кластеров
├── quality_check.py         # Оценка качества (intra − inter IF)
├── mcool_converter.py       # .mcool → текстовая матрица
├── comparison.py            # MoC, ARI, NMI, Precision/Recall/F1
├── validation.py            # ChIP-Seq валидация + загрузка ENCODE
├── visualization.py         # Контактные карты, статистика
├── utils.py                 # BED-конвертер, хелперы
└── README.md

data/
├── 4DNFI52OLNJ4.mcool
└── 4DNFIHFX73VQ.mcool
```

---

## Быстрый старт

### CLI (одиночный запуск)

```bash
# HDBSCAN на конкретном файле
python run.py single data/4DNFI52OLNJ4.mcool --chr chr19 --res 50000

# OPTICS
python run.py single data/4DNFI52OLNJ4.mcool --chr chr19 --res 50000 --method OPTICS

# Сравнение HDBSCAN vs OPTICS
python run.py compare data/4DNFI52OLNJ4.mcool --chr chr19 --res 50000

# Полный анализ с ChIP-Seq валидацией
python run.py full data/4DNFI52OLNJ4.mcool --chr chr19 --res 50000 --genome hg19
```

### Python API

```python
from pipeline import run_coitad, run_comparison, run_full_analysis

run_coitad("data/4DNFI52OLNJ4.mcool", chromosome="chr19", resolution=50000)
run_comparison("data/4DNFIHFX73VQ.mcool", chromosome="chr17", resolution=50000)
run_full_analysis("data/4DNFI52OLNJ4.mcool", chromosome="chr19", genome="hg19")
```

---

## Пакетный запуск (несколько датасетов)

### Конфигурация

Отредактируйте `run_batch.py`:

```python
DATASETS = {
    "H1_DpnII":   "data/4DNFI52OLNJ4.mcool",
    "H1_MboI":    "data/4DNFIHFX73VQ.mcool",
}
CHROMOSOMES  = ["chr19", "chr17", "chr22"]
RESOLUTIONS  = [25000, 50000, 100000]
METHODS      = ["HDBSCAN", "OPTICS"]
```

### Запуск

```bash
# Полный батч (пропускает уже посчитанное)
python run_batch.py

# Визуализация (без пересчёта)
python visualize_batch.py
```

### Что происходит

1. ChIP-Seq данные скачиваются один раз
2. Для каждой комбинации (датасет × хромосома × разрешение × метод):
   - конвертация `.mcool` → матрица (пропуск если есть)
   - генерация признаков (пропуск если есть)
   - кластеризация + извлечение TAD (пропуск если Quality/Readme.txt есть)
   - расчёт ChIP-Seq обогащения
3. Попарное сравнение HDBSCAN vs OPTICS
4. Агрегированные таблицы и отчёт

### Ожидаемое время (на датасет)

| Условие | ~Время |
|---|---|
| chr22 @ 100kb | 1–3 мин |
| chr19 @ 50kb  | 5–15 мин |
| chr17 @ 25kb  | 15–40 мин |
| chr1 @ 25kb   | 1–3 часа |

---

## Тюнинг гиперпараметров OPTICS

```bash
# Быстрый тюнинг на одной хромосоме
python tune_optics.py --mode simple

# Кросс-хромосомная валидация (защита от переобучения)
python tune_optics.py --mode cv

# Повторный запуск — перестраивает только отчёт
python tune_optics.py --mode simple
```

Тюнинг автоматически пропускает уже посчитанные конфигурации.
Результат — рекомендованные `min_samples`, `xi`, `min_cluster_size`.

---

## Визуализация

```bash
python visualize_batch.py
```

### Для каждого условия (датасет × chr × res)

| Визуализация | Описание |
|---|---|
| `*_overlay.png` | Контактная карта, оба метода наложены (синий/красный) |
| `*_optics_only.png` | TAD уникальные для OPTICS (красным), общие (серым), HDBSCAN-only (синим) |
| `*_optics_only_zoom.png` | Увеличенные панели каждого уникального OPTICS TAD |

### Агрегированные

| Визуализация | Описание |
|---|---|
| `summary_tad_counts.png` | TAD counts по всем условиям |
| `size_distributions_*.png` | Гистограммы размеров по хромосомам |
| `summary_enrichment.png` | ChIP-Seq обогащение (среднее) |
| `summary_concordance.png` | MoC / ARI / NMI heatmap |

---

## Выходная структура

```
batch_results/
├── chipseq_data/                                # ENCODE ChIP-Seq (общие)
│
├── H1_DpnII/                                    # По датасету
│   ├── chr19_50kb/
│   │   ├── data/chr19_50kb.hic
│   │   ├── features_HDBSCAN/
│   │   ├── features_OPTICS/
│   │   ├── results_HDBSCAN/TADs/ + Quality/
│   │   ├── results_OPTICS/TADs/ + Quality/
│   │   └── validation/
│   ├── chr17_50kb/
│   └── ...
│
├── H1_MboI/                                     # Второй датасет
│   └── ...
│
├── all_results.csv                              # Сводная таблица
├── all_comparisons.csv                          # HDBSCAN vs OPTICS
├── aggregate_report.txt                         # Текстовый отчёт
│
└── visualizations/
    ├── H1_DpnII/
    │   ├── chr19_50kb/
    │   │   ├── chr19_50kb_overlay.png
    │   │   ├── chr19_50kb_optics_only.png
    │   │   └── chr19_50kb_optics_only_zoom.png
    │   └── ...
    ├── H1_MboI/
    │   └── ...
    ├── summary_tad_counts.png
    ├── summary_enrichment.png
    └── summary_concordance.png
```

---

## Как устроен алгоритм

```
.mcool ──► mcool_converter ──► матрица контактов
                                      │
                              feature_generation
                           (circle of influence)
                                      │
                            ┌─────────┴─────────┐
                            ▼                   ▼
                         HDBSCAN             OPTICS
                            │                   │
                            └─────────┬─────────┘
                                      ▼
                                 extract_tad ──► quality_check
                                      │
                        ┌─────────────┼─────────────┐
                        ▼             ▼             ▼
                  comparison     validation    visualization
```

---

## Ключевые параметры

| Параметр | По умолчанию | Описание |
|---|---|---|
| `resolution` | 50000 | Разрешение Hi-C (bp) |
| `max_tad_size` | 800000 | Максимальный размер TAD (bp) |
| `min_samples` (OPTICS) | 15 | Мин. точек в окрестности |
| `xi` (OPTICS) | 0.05 | Крутизна reachability-кривой |
| `min_cluster_size` (OPTICS) | 0.05 | Мин. размер кластера (доля) |
| `tolerance` (сравнение) | 2 | Допуск на совпадение границ (бины) |

---