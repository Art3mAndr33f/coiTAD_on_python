# download_chipseq_data.py
"""
Helper script to download ChIP-Seq data from various sources
Вспомогательный скрипт для загрузки ChIP-Seq данных
"""

import requests
import gzip
import shutil
from pathlib import Path
from typing import Dict, List


class ChIPSeqDownloader:
    """Download ChIP-Seq data from public repositories"""
    
    def __init__(self, output_dir: str = "chipseq_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def download_encode_file(self, url: str, output_file: str) -> bool:
        """Download file from ENCODE"""
        try:
            print(f"Downloading from {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            output_path = self.output_dir / output_file
            
            # Save file
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Decompress if gzipped
            if output_file.endswith('.gz'):
                print(f"Decompressing...")
                with gzip.open(output_path, 'rb') as f_in:
                    with open(output_path.with_suffix(''), 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                output_path.unlink()  # Remove .gz file
                output_path = output_path.with_suffix('')
            
            print(f"Saved to {output_path}")
            return True
            
        except Exception as e:
            print(f"Error downloading: {e}")
            return False
    
    def download_hesc_markers(self, genome: str = 'hg19') -> Dict[str, str]:
        """
        Download hESC ChIP-Seq data from ENCODE
        
        Returns:
            Dict mapping marker names to downloaded file paths
        """
        base_url = f"https://hgdownload.soe.ucsc.edu/goldenPath/{genome}/encodeDCC/wgEncodeBroadHistone"
        
        tracks = {
            'CTCF': 'wgEncodeBroadHistoneH1hescCtcfStdPk.broadPeak.gz',
            'H3K4me1': 'wgEncodeBroadHistoneH1hescH3k4me1StdPk.broadPeak.gz',
            'H3K4me3': 'wgEncodeBroadHistoneH1hescH3k4me3StdPk.broadPeak.gz',
            'H3K27ac': 'wgEncodeBroadHistoneH1hescH3k27acStdPk.broadPeak.gz',
        }
        
        downloaded = {}
        
        for marker, filename in tracks.items():
            url = f"{base_url}/{filename}"
            output_file = f"{marker}_{genome}.bed"
            
            if self.download_encode_file(url, filename):
                downloaded[marker] = str(self.output_dir / output_file)
        
        # Try to download RNA Pol II from different source
        print("\nAttempting to download RNA Pol II data...")
        rnapii_url = f"https://hgdownload.soe.ucsc.edu/goldenPath/{genome}/encodeDCC/wgEncodeSydhTfbs/wgEncodeSydhTfbsH1hescPol2StdPk.narrowPeak.gz"
        if self.download_encode_file(rnapii_url, 'RNAPII.narrowPeak.gz'):
            downloaded['RNAPII'] = str(self.output_dir / 'RNAPII.bed')
        
        return downloaded


def prepare_chipseq_data(genome: str = 'hg19', output_dir: str = 'chipseq_data') -> Dict[str, str]:
    """
    Convenient function to download all necessary ChIP-Seq data
    
    Returns:
        Dict mapping marker names to file paths
    """
    downloader = ChIPSeqDownloader(output_dir)
    marker_files = downloader.download_hesc_markers(genome)
    
    print(f"\nDownloaded {len(marker_files)} marker datasets:")
    for marker, filepath in marker_files.items():
        print(f"  {marker}: {filepath}")
    
    return marker_files


if __name__ == "__main__":
    # Download hESC ChIP-Seq data
    marker_files = prepare_chipseq_data(genome='hg19', output_dir='chipseq_data')