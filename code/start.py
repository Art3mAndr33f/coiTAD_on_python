from coitad_main import CoiTAD

# Initialize coiTAD
coitad = CoiTAD(
    filepath="./data/",
    feature_filepath="./features/",
    filename="chr19_40kb.hic",
    resolution=40000,          # 40kb resolution
    max_tad_size=800000        # 800kb max TAD size
)

# Run analysis
coitad.run()