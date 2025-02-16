# PNI Model Framework
## Introduction
This study aims to address the complex challenges in predicting protein-nucleic acid interactions (PNIs), which are fundamental to various biological processes such as gene expression regulation, DNA replication, and RNA processing. Accurate prediction of PNIs is critical for understanding molecular mechanisms underlying cellular functions and diseases, including cancer and genetic disorders. Despite recent advancements, existing computational methods often struggle with capturing intricate patterns and long-range dependencies in sequence data, limiting their utility in real-world applications.

To overcome these challenges, this research introduces a novel multi-task deep learning framework, integrating models such as PNI-FCN, PNI-Transformer, and PNI-MAMBA(s). The work is significant for its incorporation of binding site attention mechanisms and multi-task learning to enhance both interpretability and prediction accuracy. The proposed framework outperforms existing methods in terms of accuracy, robustness, and computational efficiency, as demonstrated through rigorous validation on merged DNA and RNA datasets, simulated novel data, and high-throughput screening scenarios.

By providing accurate and interpretable predictions of PNIs, this study not only advances our understanding of fundamental biological processes but also lays a foundation for developing targeted therapeutic strategies. Its high computational efficiency further enables practical applications in high-throughput screening, accelerating research in drug discovery and molecular biology.

This project implements a predictive model framework (PNI) for protein-nucleic acid interaction prediction. Users can switch between different models (FCN, Transformer, MAMBA, MAMBA2) by adjusting the parameter settings. It encodes both ligand and peptide sequences from the input file and performs predictions. The final prediction results will be saved in the result directory as .pkl files.

## Usage
### 1. Select a Model
Inside the if __name__ == '__main__': block, uncomment the parameter block corresponding to your desired model and comment out the others. For example:
```python
# Select the PNI_FCN model
params = dict(
    base='FCN',
    ...
)
```
### 2. Input Data
Ensure the input file is located at ```data/BioLip_20240530/BioLip_20240530_candidate.csv.```

### 3. Run the Script
Execute the script using the following command:
```python
python main.py
```

### 4. Output
After the predictions are completed, the results will be saved as .pkl files in the result directory.

