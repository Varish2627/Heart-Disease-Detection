# Heart Disease Detection

This project implements heart disease detection using machine learning models on multiple datasets.

## Datasets

The project uses the following datasets (included in the `Datasets/` folder):
- Cardiovascular Heart Disease Dataset.csv
- heart_cleveland_upload.csv

## Requirements

Install the following Python packages:

```bash
pip install numpy
pip install pandas
pip install tqdm
pip install matplotlib
pip install scikit-learn
pip install seaborn
pip install torch
pip install torchvision torchaudio
```

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Varish2627/Heart-Disease-Detection.git
   cd Heart-Disease-Detection
   ```

2. Install the required packages as listed above.

3. Run the main script:
   ```bash
   python Main.py
   ```

Running `Main.py` will automatically process all datasets and generate results, including graphs and metrics, saved in the `results/` folder.

## Project Structure

- `Main.py`: Main script for Cardiovascular dataset processing.
- `cleveland.py`: Script for Cleveland dataset processing.
- `Cardiovascular_graph.py`: Graph generation for Cardiovascular dataset.
- `cleveland_graph.py`: Graph generation for Cleveland dataset.
- `Datasets/`: Folder containing the dataset files.
- `results/`: Folder where output graphs and metrics are saved.

## Results

The scripts will generate various visualizations and performance metrics for heart disease prediction models.