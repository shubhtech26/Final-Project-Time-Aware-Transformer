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

## Modifications for macOS

The code has been modified to run efficiently on macOS:
- Sampled a smaller subset of data (5% of orders)
- Reduced model size (hidden_size: 32, num_heads: 2)
- Decreased training epochs (5 epochs)
- Smaller batch size (16)
- Ablation study is disabled by default to save time/memory

## Results

The experiment outputs will be saved to the `output/` directory:
- Model checkpoints
- Performance metrics and comparison charts
- Time segment analysis results

## Model Details

The project implements two models:
1. **SASRec (Baseline)**: Self-Attentive Sequential Recommendation model that only uses item sequences
2. **TimeSASRec**: Time-Aware extension that incorporates temporal features like:
   - Hour of day
   - Day of week
   - Days since prior order
   - Cyclic time encoding

## Notes

- If you want to run the full experiment, modify the sampling rate in the `main()` function
- To enable the ablation study, uncomment the relevant code in the main function
- Adjust batch sizes and model configurations based on your system's capabilities 