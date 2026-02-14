import cooler
import numpy as np
from pathlib import Path


class McoolConverter:
    """Конвертер .mcool файлов в текстовые матрицы для coiTAD"""
    
    def __init__(self, mcool_file: str):
        """
        Инициализация конвертера
        
        Args:
            mcool_file: Путь к .mcool файлу
        """
        self.mcool_file = mcool_file
        
    def list_resolutions(self):
        """Показать доступные разрешения в .mcool файле"""
        try:
            resolutions = cooler.fileops.list_coolers(self.mcool_file)
            print("Доступные разрешения:")
            for res in resolutions:
                print(f"  - {res}")
            return resolutions
        except Exception as e:
            print(f"Ошибка при чтении файла: {e}")
            return []
    
    def extract_chromosome(self, 
                          chromosome: str, 
                          resolution: int,
                          output_file: str = None,
                          normalize: bool = True,
                          balance: bool = True) -> np.ndarray:
        """
        Извлечь контактную матрицу для конкретной хромосомы
        
        Args:
            chromosome: Название хромосомы (например, 'chr19', '19')
            resolution: Разрешение в bp (например, 50000 для 50kb)
            output_file: Путь для сохранения (опционально)
            normalize: Нормализовать матрицу
            balance: Использовать balanced данные (ICE нормализация)
            
        Returns:
            Контактная матрица как numpy array
        """
        # Формируем URI для нужного разрешения
        uri = f"{self.mcool_file}::resolutions/{resolution}"
        
        try:
            # Загружаем cooler объект
            clr = cooler.Cooler(uri)
            
            # Проверяем доступные хромосомы
            available_chroms = clr.chromnames
            
            # Добавляем 'chr' префикс если нужно
            if chromosome not in available_chroms:
                if f'chr{chromosome}' in available_chroms:
                    chromosome = f'chr{chromosome}'
                elif chromosome.replace('chr', '') in available_chroms:
                    chromosome = chromosome.replace('chr', '')
                else:
                    raise ValueError(f"Хромосома {chromosome} не найдена. "
                                   f"Доступные: {available_chroms}")
            
            print(f"Извлечение {chromosome} с разрешением {resolution}bp...")
            
            # Извлекаем матрицу
            if balance and 'weight' in clr.bins().columns:
                # Используем balanced (ICE-нормализованные) данные
                matrix = clr.matrix(balance=True).fetch(chromosome)
            else:
                # Используем raw данные
                matrix = clr.matrix(balance=False).fetch(chromosome)
            
            # Обработка NaN значений
            matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Дополнительная нормализация если требуется
            if normalize and not balance:
                # Простая нормализация по максимуму
                max_val = np.max(matrix)
                if max_val > 0:
                    matrix = matrix / max_val
            
            print(f"Размер матрицы: {matrix.shape}")
            print(f"Диапазон значений: [{np.min(matrix):.6f}, {np.max(matrix):.6f}]")
            print(f"Ненулевых элементов: {np.count_nonzero(matrix)} "
                  f"({100*np.count_nonzero(matrix)/matrix.size:.2f}%)")
            
            # Сохраняем в текстовый файл
            if output_file:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                print(f"Сохранение в {output_file}...")
                np.savetxt(output_file, matrix, fmt='%.6f')
                print("Сохранено успешно!")
            
            return matrix
            
        except Exception as e:
            print(f"Ошибка при извлечении данных: {e}")
            raise
    
    def extract_region(self,
                      chromosome: str,
                      start: int,
                      end: int,
                      resolution: int,
                      output_file: str = None,
                      balance: bool = True) -> np.ndarray:
        """
        Извлечь контактную матрицу для региона хромосомы
        
        Args:
            chromosome: Название хромосомы
            start: Начальная позиция (bp)
            end: Конечная позиция (bp)
            resolution: Разрешение в bp
            output_file: Путь для сохранения
            balance: Использовать balanced данные
            
        Returns:
            Контактная матрица региона
        """
        uri = f"{self.mcool_file}::resolutions/{resolution}"
        
        try:
            clr = cooler.Cooler(uri)
            
            # Формируем region string
            region = f"{chromosome}:{start}-{end}"
            
            print(f"Извлечение региона {region} с разрешением {resolution}bp...")
            
            if balance and 'weight' in clr.bins().columns:
                matrix = clr.matrix(balance=True).fetch(region)
            else:
                matrix = clr.matrix(balance=False).fetch(region)
            
            matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
            
            print(f"Размер матрицы: {matrix.shape}")
            
            if output_file:
                np.savetxt(output_file, matrix, fmt='%.6f')
                print(f"Сохранено в {output_file}")
            
            return matrix
            
        except Exception as e:
            print(f"Ошибка: {e}")
            raise
    
    def get_chromosome_info(self, resolution: int):
        """Получить информацию о хромосомах"""
        uri = f"{self.mcool_file}::resolutions/{resolution}"
        
        try:
            clr = cooler.Cooler(uri)
            chromsizes = clr.chromsizes
            
            print("\nИнформация о хромосомах:")
            print(f"{'Хромосома':<15} {'Размер (bp)':<15} {'Bins':<10}")
            print("-" * 40)
            
            for chrom, size in chromsizes.items():
                n_bins = size // resolution
                print(f"{chrom:<15} {size:<15,} {n_bins:<10,}")
            
            return chromsizes
            
        except Exception as e:
            print(f"Ошибка: {e}")
            return None


def main():
    """Пример использования"""
    
    # 1. Создаем конвертер
    converter = McoolConverter("4DNFI52OLNJ4.mcool")
    
    # 2. Смотрим доступные разрешения
    converter.list_resolutions()
    
    # 3. Смотрим информацию о хромосомах (для разрешения 50kb)
    converter.get_chromosome_info(resolution=50000)
    
    # 4. Извлекаем хромосому 19 с разрешением 50kb
    matrix = converter.extract_chromosome(
        chromosome="chr19",  # или просто "19"
        resolution=50000,
        output_file="data/chr19_50kb.hic",
        balance=True  # Использовать ICE-нормализованные данные
    )
    
    # 5. Опционально: извлечь только регион
    # matrix_region = converter.extract_region(
    #     chromosome="chr19",
    #     start=0,
    #     end=20000000,  # 20 Mb
    #     resolution=50000,
    #     output_file="data/chr19_region_50kb.hic"
    # )


if __name__ == "__main__":
    main()