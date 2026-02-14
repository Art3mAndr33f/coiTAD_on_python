# quick_visualize.py
from visualize_coitad import visualize_coitad_results

# Визуализация результатов
visualizer = visualize_coitad_results(
    results_dir="coitad_output/results",
    data_dir="coitad_output/data",
    chromosome="chr19",
    resolution=40000,
    best_radius=2  # Ваш recommended radius
)