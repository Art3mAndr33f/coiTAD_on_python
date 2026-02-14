"""Convert TAD results to BED format"""

import numpy as np
from pathlib import Path


def convert_to_bed(input_file: str, output_file: str, first_col: int = 2, second_col: int = 4):
    """
    Convert TAD domain file to BED format
    
    Args:
        input_file: Path to input TAD file
        output_file: Path to output BED file
        first_col: Column index for start position (1-indexed)
        second_col: Column index for end position (1-indexed)
    """
    try:
        # Read input file
        data = np.loadtxt(input_file, skiprows=1)
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        # Extract columns (convert to 0-indexed)
        starts = data[:, first_col - 1].astype(int)
        stops = data[:, second_col - 1].astype(int)
        
        # Write to BED file
        with open(output_file, 'w') as f:
            for start, stop in zip(starts, stops):
                f.write(f"{start}\t{stop}\n")
        
        print("Conversion completed successfully.")
        
    except Exception as e:
        print(f"Error during conversion: {e}")