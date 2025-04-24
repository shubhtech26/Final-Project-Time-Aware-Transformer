# Time-Aware Recommender System

This project implements a time-aware recommender system using the Instacart dataset. It compares a standard SASRec model with a time-enhanced version that incorporates temporal features.

## Dataset

This project uses the Instacart Online Grocery Basket Analysis dataset, which needs to be downloaded separately:

1. Download the dataset from Kaggle: [Instacart Online Grocery Basket Analysis Dataset](https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset)
2. Extract the downloaded files into a `Dataset` folder in the project root directory
3. Ensure the following files are present in the `Dataset` directory:
   - `orders.csv`
   - `order_products__prior.csv`
   - `products.csv`
   - `aisles.csv`
   - `departments.csv`

## Setup

1. Clone this repository:
   ```
   git clone <repository-url>
   cd <repository-name>
   ```

2. Create and activate a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the model:
   ```
   python time_aware_recommender.py
   ```

## Results

The experiment outputs will be saved to the `output/` directory:
- Model checkpoints
- Performance metrics and comparison charts
- Time segment analysis results

This repository includes key output files from previous runs in the `output/` folder:
- CSV files with metrics and comparison data
- Visualization plots showing model performance
- Time segment analysis results

## GitHub Repository Structure

```
├── Dataset/ (not included, download separately)
├── output/
│   ├── *.csv (performance metrics)
│   └── *.png (visualization charts)
├── .gitignore
├── README.md
├── requirements.txt
└── time_aware_recommender.py
```

The large model checkpoint files (*.pth) are excluded from the repository but all result files and visualizations are included to allow for easy review of the model's performance.

## Model Details

The project implements two models:
1. **SASRec (Baseline)**: Self-Attentive Sequential Recommendation model that only uses item sequences
2. **TimeSASRec**: Time-Aware extension that incorporates temporal features like:
   - Hour of day
   - Day of week
   - Days since prior order
   - Cyclic time encoding

## Notes

This implementation processes the full Instacart dataset and performs comprehensive analyses:
- Full dataset training for both baseline and time-aware models
- Detailed time segment analysis (morning/evening, weekday/weekend)
- Ablation study to understand the importance of different temporal features
- Performance comparison between models using multiple metrics (Recall@k, NDCG@k, Hit@k) 