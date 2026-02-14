from mcool_converter import McoolConverter
from coitad_main import CoiTAD
from pathlib import Path
from visualize_coitad import visualize_coitad_results


def run_coitad_from_mcool(
    mcool_file: str,
    chromosome: str = "chr19",
    resolution: int = 50000,
    output_dir: str = "coitad_results"
):
    """
    Полный pipeline: конвертация .mcool -> запуск coiTAD
    
    Args:
        mcool_file: Путь к .mcool файлу
        chromosome: Хромосома для анализа
        resolution: Разрешение в bp
        output_dir: Директория для результатов
    """
    
    # Создаем директории
    Path(output_dir).mkdir(exist_ok=True)
    Path(f"{output_dir}/data").mkdir(exist_ok=True)
    Path(f"{output_dir}/features").mkdir(exist_ok=True)
    
    # Шаг 1: Конвертация .mcool в текстовую матрицу
    print("=" * 60)
    print("Шаг 1: Конвертация .mcool файла")
    print("=" * 60)
    
    converter = McoolConverter(mcool_file)
    
    # Показываем доступные разрешения
    converter.list_resolutions()
    
    # Показываем информацию о хромосомах
    converter.get_chromosome_info(resolution)
    
    # Извлекаем данные
    output_matrix = f"{output_dir}/data/{chromosome}_{resolution//1000}kb.hic"
    matrix = converter.extract_chromosome(
        chromosome=chromosome,
        resolution=resolution,
        output_file=output_matrix,
        balance=True  # ICE нормализация
    )
    
    # Шаг 2: Запуск coiTAD
    print("\n" + "=" * 60)
    print("Шаг 2: Запуск coiTAD")
    print("=" * 60)
    
    coitad = CoiTAD(
        filepath=f"{output_dir}/data/",
        feature_filepath=f"{output_dir}/features/",
        filename=f"{chromosome}_{resolution//1000}kb.hic",
        resolution=resolution,
        max_tad_size=800000,
        output_folder=f"{output_dir}/results"
    )
    
    coitad.run()

    print("\nГенерация визуализаций...")
    visualize_coitad_results(
        results_dir=f"{output_dir}/results",
        data_dir=f"{output_dir}/data",
        chromosome=chromosome,
        resolution=resolution,
        best_radius=coitad.best_radius
    )
    
    print("\n" + "=" * 60)
    print("Анализ завершен!")
    print("=" * 60)
    print(f"Результаты сохранены в: {output_dir}/results/")
    print(f"Рекомендуемый radius: {coitad.best_radius}")


if __name__ == "__main__":
    # Пример запуска
    run_coitad_from_mcool(
        mcool_file="4DNFI52OLNJ4.mcool",
        chromosome="chr19",  # или "19"
        resolution=50000,     # 50kb
        output_dir="coitad_output"
    )
