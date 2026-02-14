# visualize_coitad.py
"""
Визуализация результатов coiTAD
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
from pathlib import Path
from typing import List, Tuple, Optional
import seaborn as sns


class CoiTADVisualizer:
    """Класс для визуализации результатов coiTAD"""
    
    def __init__(self, 
                 contact_matrix_file: str,
                 tad_file: str,
                 resolution: int = 40000,
                 output_dir: str = "visualizations"):
        """
        Инициализация визуализатора
        
        Args:
            contact_matrix_file: Путь к файлу контактной матрицы
            tad_file: Путь к файлу с TAD границами
            resolution: Разрешение данных в bp
            output_dir: Директория для сохранения визуализаций
        """
        self.contact_matrix = np.loadtxt(contact_matrix_file)
        self.tad_borders = self.load_tad_borders(tad_file)
        self.resolution = resolution
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def load_tad_borders(self, tad_file: str) -> List[Tuple[int, int]]:
        """Загрузка TAD границ из файла"""
        try:
            # Пробуем загрузить как _TAD_BinID.txt
            data = np.loadtxt(tad_file, dtype=int)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            return [(row[0], row[1]) for row in data]
        except:
            # Пробуем загрузить как _domain.txt с заголовком
            data = np.loadtxt(tad_file, skiprows=1, usecols=(0, 2), dtype=int)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            return [(row[0], row[1]) for row in data]
    
    def plot_contact_map_with_tads(self,
                                   figsize: Tuple[int, int] = (12, 10),
                                   cmap: str = 'Reds',
                                   log_scale: bool = True,
                                   vmin: Optional[float] = None,
                                   vmax: Optional[float] = None,
                                   title: str = None,
                                   save_name: str = "contact_map_with_tads.png",
                                   dpi: int = 300):
        """
        Визуализация контактной карты с TAD границами
        
        Args:
            figsize: Размер фигуры
            cmap: Цветовая схема
            log_scale: Использовать логарифмическую шкалу
            vmin, vmax: Пределы цветовой шкалы
            title: Заголовок графика
            save_name: Имя файла для сохранения
            dpi: Разрешение изображения
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Отображение контактной карты
        matrix = self.contact_matrix.copy()
        matrix[matrix == 0] = np.nan  # Нули как NaN для лучшей визуализации
        
        if log_scale:
            im = ax.imshow(matrix, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax),
                          interpolation='none', origin='upper')
        else:
            im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax,
                          interpolation='none', origin='upper')
        
        # Добавление TAD границ
        self._draw_tad_rectangles(ax, self.tad_borders, 
                                 color='blue', linewidth=2, alpha=0.8)
        
        # Настройка осей
        n_bins = matrix.shape[0]
        self._setup_axes(ax, n_bins)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Contact Frequency', rotation=270, labelpad=20, fontsize=12)
        
        # Заголовок
        if title is None:
            title = f'Hi-C Contact Map with TAD Boundaries\n({len(self.tad_borders)} TADs detected)'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Сохранение
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Сохранено: {output_path}")
        plt.show()
        
    def plot_tad_zoom(self,
                     tad_index: int = 0,
                     context_bins: int = 10,
                     figsize: Tuple[int, int] = (10, 8),
                     cmap: str = 'RdYlBu_r',
                     save_name: str = None):
        """
        Увеличенный вид отдельного TAD с контекстом
        
        Args:
            tad_index: Индекс TAD для визуализации (0-based)
            context_bins: Количество bins вокруг TAD для контекста
            figsize: Размер фигуры
            cmap: Цветовая схема
            save_name: Имя файла для сохранения
        """
        if tad_index >= len(self.tad_borders):
            print(f"TAD индекс {tad_index} вне диапазона. Всего TADs: {len(self.tad_borders)}")
            return
        
        start, end = self.tad_borders[tad_index]
        
        # Добавляем контекст
        start_ctx = max(0, start - context_bins)
        end_ctx = min(self.contact_matrix.shape[0], end + context_bins)
        
        # Извлекаем подматрицу
        submatrix = self.contact_matrix[start_ctx:end_ctx, start_ctx:end_ctx]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Отображение
        im = ax.imshow(submatrix, cmap=cmap, interpolation='none', origin='upper')
        
        # Рисуем границу TAD (относительно подматрицы)
        rel_start = start - start_ctx
        rel_end = end - start_ctx
        rect = patches.Rectangle((rel_start-0.5, rel_start-0.5),
                                 rel_end - rel_start + 1,
                                 rel_end - rel_start + 1,
                                 linewidth=3, edgecolor='red',
                                 facecolor='none', linestyle='--')
        ax.add_patch(rect)
        
        # Настройка осей
        ax.set_xlabel(f'Genomic Position (bin)', fontsize=12)
        ax.set_ylabel(f'Genomic Position (bin)', fontsize=12)
        
        # Заголовок
        tad_size_kb = (end - start) * self.resolution / 1000
        title = f'TAD #{tad_index + 1}\n'
        title += f'Position: {start}-{end} (bins) | Size: {tad_size_kb:.1f} kb'
        ax.set_title(title, fontsize=13, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Contact Frequency', rotation=270, labelpad=15)
        
        plt.tight_layout()
        
        # Сохранение
        if save_name is None:
            save_name = f"tad_{tad_index}_zoom.png"
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Сохранено: {output_path}")
        plt.show()
    
    def plot_tad_statistics(self, 
                           figsize: Tuple[int, int] = (14, 5),
                           save_name: str = "tad_statistics.png"):
        """
        Визуализация статистики TADs
        """
        # Вычисляем размеры TADs
        tad_sizes = [(end - start) * self.resolution / 1000 
                     for start, end in self.tad_borders]  # в kb
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # 1. Гистограмма размеров TAD
        axes[0].hist(tad_sizes, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(np.median(tad_sizes), color='red', linestyle='--', 
                       linewidth=2, label=f'Median: {np.median(tad_sizes):.1f} kb')
        axes[0].set_xlabel('TAD Size (kb)', fontsize=11)
        axes[0].set_ylabel('Frequency', fontsize=11)
        axes[0].set_title('TAD Size Distribution', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # 2. Boxplot размеров TAD
        bp = axes[1].boxplot(tad_sizes, vert=True, patch_artist=True,
                            boxprops=dict(facecolor='lightblue'),
                            medianprops=dict(color='red', linewidth=2))
        axes[1].set_ylabel('TAD Size (kb)', fontsize=11)
        axes[1].set_title('TAD Size Statistics', fontsize=12, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        # Добавляем статистику
        stats_text = f'Mean: {np.mean(tad_sizes):.1f} kb\n'
        stats_text += f'Median: {np.median(tad_sizes):.1f} kb\n'
        stats_text += f'Std: {np.std(tad_sizes):.1f} kb\n'
        stats_text += f'Min: {np.min(tad_sizes):.1f} kb\n'
        stats_text += f'Max: {np.max(tad_sizes):.1f} kb'
        axes[1].text(1.15, np.median(tad_sizes), stats_text,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=9)
        
        # 3. TAD coverage вдоль хромосомы
        n_bins = self.contact_matrix.shape[0]
        coverage = np.zeros(n_bins)
        for start, end in self.tad_borders:
            coverage[start:end+1] = 1
        
        genomic_pos = np.arange(n_bins) * self.resolution / 1_000_000  # в Mb
        axes[2].fill_between(genomic_pos, 0, coverage, alpha=0.5, color='green')
        axes[2].set_xlabel('Genomic Position (Mb)', fontsize=11)
        axes[2].set_ylabel('TAD Coverage', fontsize=11)
        axes[2].set_title('TAD Coverage Along Chromosome', fontsize=12, fontweight='bold')
        axes[2].set_ylim(-0.1, 1.1)
        axes[2].grid(axis='y', alpha=0.3)
        
        # Добавляем общую статистику
        coverage_pct = np.sum(coverage) / n_bins * 100
        axes[2].text(0.02, 0.98, f'Coverage: {coverage_pct:.1f}%\nTotal TADs: {len(self.tad_borders)}',
                    transform=axes[2].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=9)
        
        plt.tight_layout()
        
        # Сохранение
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Сохранено: {output_path}")
        plt.show()
        
    def plot_tad_heatmap_annotated(self,
                                   max_tads: int = 20,
                                   figsize: Tuple[int, int] = (14, 12),
                                   cmap: str = 'YlOrRd',
                                   save_name: str = "tad_heatmap_annotated.png"):
        """
        Контактная карта с пронумерованными TAD
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Отображение контактной карты
        matrix = self.contact_matrix.copy()
        matrix[matrix == 0] = np.nan
        
        im = ax.imshow(matrix, cmap=cmap, norm=LogNorm(),
                      interpolation='none', origin='upper', aspect='auto')
        
        # Рисуем и нумеруем TAD
        n_display = min(len(self.tad_borders), max_tads)
        for i, (start, end) in enumerate(self.tad_borders[:n_display]):
            # Прямоугольник
            self._draw_tad_rectangles(ax, [(start, end)], 
                                     color='blue', linewidth=2, alpha=0.7)
            
            # Номер TAD
            center = (start + end) / 2
            ax.text(center, center, str(i+1),
                   ha='center', va='center',
                   fontsize=10, fontweight='bold',
                   color='white',
                   bbox=dict(boxstyle='circle', facecolor='blue', alpha=0.8))
        
        # Настройка осей
        n_bins = matrix.shape[0]
        self._setup_axes(ax, n_bins)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Contact Frequency (log)', rotation=270, labelpad=20)
        
        # Заголовок
        title = f'Hi-C Contact Map with Numbered TADs\n'
        title += f'(Showing first {n_display} of {len(self.tad_borders)} TADs)'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Сохранение
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Сохранено: {output_path}")
        plt.show()
    
    def plot_diagonal_view(self,
                          window_size: int = 50,
                          figsize: Tuple[int, int] = (14, 6),
                          save_name: str = "diagonal_view.png"):
        """
        Визуализация диагонального окна контактной карты
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Создаем диагональную маску
        n = self.contact_matrix.shape[0]
        masked_matrix = np.zeros_like(self.contact_matrix)
        
        for i in range(n):
            for j in range(max(0, i-window_size), min(n, i+window_size+1)):
                masked_matrix[i, j] = self.contact_matrix[i, j]
        
        masked_matrix[masked_matrix == 0] = np.nan
        
        # Отображение
        im = ax.imshow(masked_matrix, cmap='RdPu', norm=LogNorm(),
                      interpolation='none', origin='upper')
        
        # Добавляем TAD границы
        self._draw_tad_rectangles(ax, self.tad_borders, 
                                 color='cyan', linewidth=1.5, alpha=0.9)
        
        # Настройка
        self._setup_axes(ax, n)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Contact Frequency', rotation=270, labelpad=15)
        
        ax.set_title(f'Diagonal View (±{window_size} bins) with TAD Boundaries',
                    fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Сохранено: {output_path}")
        plt.show()
    
    def plot_comparison_subplots(self,
                                tad_indices: List[int] = None,
                                n_cols: int = 3,
                                figsize_per_subplot: Tuple[int, int] = (4, 4),
                                save_name: str = "tad_comparison.png"):
        """
        Сравнение нескольких TAD в виде subplot-ов
        """
        if tad_indices is None:
            # Выбираем первые 6 TAD
            tad_indices = list(range(min(6, len(self.tad_borders))))
        
        n_tads = len(tad_indices)
        n_rows = int(np.ceil(n_tads / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, 
                                figsize=(figsize_per_subplot[0]*n_cols, 
                                        figsize_per_subplot[1]*n_rows))
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, tad_idx in enumerate(tad_indices):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            if tad_idx >= len(self.tad_borders):
                ax.axis('off')
                continue
            
            start, end = self.tad_borders[tad_idx]
            context = 5
            start_ctx = max(0, start - context)
            end_ctx = min(self.contact_matrix.shape[0], end + context)
            
            submatrix = self.contact_matrix[start_ctx:end_ctx, start_ctx:end_ctx]
            
            im = ax.imshow(submatrix, cmap='YlOrRd', interpolation='none')
            
            # TAD граница
            rel_start = start - start_ctx
            rel_end = end - start_ctx
            rect = patches.Rectangle((rel_start-0.5, rel_start-0.5),
                                     rel_end - rel_start + 1,
                                     rel_end - rel_start + 1,
                                     linewidth=2, edgecolor='blue',
                                     facecolor='none')
            ax.add_patch(rect)
            
            tad_size_kb = (end - start) * self.resolution / 1000
            ax.set_title(f'TAD #{tad_idx+1} ({tad_size_kb:.0f} kb)', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Скрываем лишние subplot-ы
        for idx in range(n_tads, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        plt.suptitle('TAD Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        output_path = self.output_dir / save_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Сохранено: {output_path}")
        plt.show()
    
    def generate_all_plots(self):
        """Генерация всех визуализаций"""
        print("Генерация всех визуализаций...")
        print("\n1. Контактная карта с TAD границами...")
        self.plot_contact_map_with_tads()
        
        print("\n2. Статистика TAD...")
        self.plot_tad_statistics()
        
        print("\n3. Аннотированная карта...")
        self.plot_tad_heatmap_annotated()
        
        print("\n4. Диагональный вид...")
        self.plot_diagonal_view()
        
        print("\n5. Увеличенный вид первого TAD...")
        if len(self.tad_borders) > 0:
            self.plot_tad_zoom(0)
        
        print("\n6. Сравнение TAD...")
        self.plot_comparison_subplots()
        
        print(f"\n✓ Все визуализации сохранены в: {self.output_dir}/")
    
    # Вспомогательные методы
    def _draw_tad_rectangles(self, ax, borders, color='blue', linewidth=2, alpha=0.8):
        """Рисование прямоугольников TAD"""
        for start, end in borders:
            rect = patches.Rectangle(
                (start - 0.5, start - 0.5),
                end - start + 1,
                end - start + 1,
                linewidth=linewidth,
                edgecolor=color,
                facecolor='none',
                alpha=alpha
            )
            ax.add_patch(rect)
    
    def _setup_axes(self, ax, n_bins):
        """Настройка осей с координатами"""
        # Определяем тиковые позиции
        step = max(1, n_bins // 10)
        tick_positions = np.arange(0, n_bins, step)
        tick_labels = [f'{int(pos * self.resolution / 1_000_000):.1f}' 
                      for pos in tick_positions]
        
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=10)
        ax.set_yticklabels(tick_labels, fontsize=10)
        
        ax.set_xlabel('Genomic Position (Mb)', fontsize=12)
        ax.set_ylabel('Genomic Position (Mb)', fontsize=12)
        
        ax.grid(False)


# Использование
def visualize_coitad_results(
    results_dir: str = "coitad_output/results",
    data_dir: str = "coitad_output/data",
    chromosome: str = "chr19",
    resolution: int = 40000,
    best_radius: int = 2
):
    """
    Главная функция для визуализации результатов coiTAD
    
    Args:
        results_dir: Директория с результатами coiTAD
        data_dir: Директория с исходными данными
        chromosome: Название хромосомы
        resolution: Разрешение
        best_radius: Лучший radius из анализа
    """
    # Формируем пути к файлам
    contact_matrix_file = f"{data_dir}/{chromosome}_{resolution//1000}kb.hic"
    tad_file = f"{results_dir}/TADs/HDBSCAN_{best_radius}_TAD_BinID.txt"
    
    print(f"Загрузка данных...")
    print(f"  Контактная матрица: {contact_matrix_file}")
    print(f"  TAD файл: {tad_file}")
    
    # Создаем визуализатор
    visualizer = CoiTADVisualizer(
        contact_matrix_file=contact_matrix_file,
        tad_file=tad_file,
        resolution=resolution,
        output_dir=f"{results_dir}/visualizations"
    )
    
    # Генерируем все визуализации
    visualizer.generate_all_plots()
    
    return visualizer


# Пример использования
if __name__ == "__main__":
    # После запуска coiTAD
    visualizer = visualize_coitad_results(
        results_dir="coitad_output/results",
        data_dir="coitad_output/data",
        chromosome="chr19",
        resolution=40000,
        best_radius=2  # Используйте ваш recommended radius
    )
    
    # Или создайте отдельные визуализации:
    # visualizer.plot_contact_map_with_tads()
    # visualizer.plot_tad_zoom(tad_index=0)
    # visualizer.plot_tad_statistics()
