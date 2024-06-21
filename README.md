# Two-step Trading System for Cryptocurrency Markets

> “I always say the beauty is in the mix.” – Nancy Pelosi

This repository is a public version with description about trading system for cryptocurrency markets. It does not contain full project.

## Acknowledgements
I would like to express my deepest gratitude to my advisor, [Andrew](https://github.com/5x12), whose insightful feedback and support were instrumental in the development of this project. Without his guidance, this project would not have been as successful and enjoyable.

## Overview
This project was developed as part of my Master's thesis at HSE. The aim is to create an automated trading system capable of generating and validating signals and executing orders on preferred cryptocurrency exchanges.

The system employs various strategies to generate buy/sell signals. These buy signals are then validated using a classification model to filter out false signals.

## Data Acquisition
The system utilizes default OHLCV data, accessible via API from any exchange in a unified format. The main period analyzed spans from January 1, 2021, to August 29, 2023, covering different market conditions (bull, bear, flat).

## Strategies Development
Developing effective trading strategies is crucial for generating profitable signals. I coded these strategies in Trading View to visualize each entry, simulate the market in real-time, and examine indicator connections. While I won't share the specific strategies used, you can create strategies that fit your trading style.

Guidance on creating trading strategies can be found on [robuxio.com/blog](https://robuxio.com/blog).

## Step 1: Generating Signals
### Grid Search
To maximize trading profits, optimal parameters for the algorithm are found using GridSearch over a network of different parameter combinations for each strategy input. Key metrics for evaluating parameters include:
- ROI
- Average profit per trade
- Number of trades
- Custom metric: ROI multiplied by average profit per trade

Evaluating more parameter combinations during GridSearch can enhance potential performance.

### Parameter Validation
After identifying optimal trading parameters, they need validation. GridSearch focuses on certain metrics, but these combinations also require examination for their money-generating potential. Important metrics include:

| Metric             | Meaning                             | Formula                                                                                 |
|--------------------|-------------------------------------|-----------------------------------------------------------------------------------------|
| ROI                | Total ROI per simulation            | balance value at the end / initial balance                                              |
| Profit             | Total profit in USD per simulation  | balance value at the end - initial balance                                              |
| # of trades        | Total number of trades executed     | count of trades                                                                         |
| Avg trades per day | Average number of trades per day    | total number of trades / number of days                                                 |
| Profit factor      | Profit per one lost dollar          | total profit from winning trades / total loss from losing trades                        |
| Max %              | Maximum percentage of profit        | maximum value of percentage among all trades                                            |
| Avg %              | Average percentage of profit        | ROI / total number of trades                                                            |
| Max DD             | Maximum drawdown in one trade       | maximum value of drawdown among all trades                                              |
| Avg DD             | Average drawdown in one trade       | sum of drawdown among all trades / total number of trades                               |
| Avg time           | Average time in position for trade  | sum of time in position (end time - start time) / total time                            |
| Max unrealized     | Maximum unrealized profit per trade | maximum value of unrealized profit among all trades                                     |
| Avg unrealized     | Average unrealized profit per trade | sum of unrealized profit among all trades / total number of trades                      |

Additionally, graphical representations include:
- Asset price
- Net incomes per transaction (long and short)
- Accumulated net income (long and short)
- Total accumulated net income
- Drawdowns

Analyzing the list of trades helps understand the conditions under which each trade was executed.

## Step 2: ML Classification
After finding optimal parameters for all strategies, the next step is applying machine learning techniques to improve the system. The ML algorithm filters out trades likely to be unprofitable based on current market conditions. 

Each strategy has a different classifier due to varying abilities in different market types. A trade is classified as unprofitable if its profit percentage is less than 1%.

### Data Preprocessing
#### Generating Features
Indicators of various types provide the model with current market state information:
- **Trend-following indicators**: Moving Average Convergence Divergence (MACD), Average Directional Index (ADX)
- **Volatility indicators**: Bollinger Bands, Average True Range (ATR), Standard Deviation (STD)
- **Support and Resistance Indicators**: Pivot Points, Fibonacci Retracement
- **Oscillators**: Relative Strength Index (RSI), Stochastic Oscillator, Commodity Channel Index (CCI), Williams %R, Rate of Change (ROC)

Indicators are represented as lagged differences and lagged percentage changes. Lags are determined using ACF-PACF analysis, typically 1, 2, 5, 10, 15, and 20.

#### Resampling
To balance the dataset, I applied SMOTE, which generates synthetic data with the same characteristics as the features. Scaling and One-Hot Encoding were also performed. The final pipeline is illustrated below:
*Insert pipeline picture*

### Basic Model Development
Given the dataset's length (533 trades), Cross-Validation was used to maximize training and testing data. Seven models were initially tested:
- Random Forest Classifier
- Logistic Regression
- XGBoost Classifier
- CatBoost Classifier
- SGD Classifier
- Support Vector Classification
- K Neighbors Classifier

Performance metrics included Accuracy, Precision, Recall, F1, and ROC-AUC. The results are summarized below:

| Model          | Accuracy | Precision | Recall | F1  | Roc-Auc |
|----------------|----------|-----------|--------|-----|---------|
| CatBoost_0     | 0.71     | 0.79      | 0.84   | 0.81| 0.69    |
| CatBoost_1     | 0.71     | 0.45      | 0.36   | 0.40| 0.69    |
| RandomForest_0 | 0.73     | 0.79      | 0.85   | 0.82| 0.67    |
| RandomForest_1 | 0.73     | 0.48      | 0.38   | 0.42| 0.67    |
| SVC_0          | 0.71     | 0.78      | 0.83   | 0.81| 0.66    |
| SVC_1          | 0.71     | 0.43      | 0.35   | 0.39| 0.66    |
| LogReg_0       | 0.64     | 0.78      | 0.71   | 0.74| 0.64    |
| LogReg_1       | 0.64     | 0.36      | 0.45   | 0.40| 0.64    |
| XGB_0          | 0.71     | 0.78      | 0.83   | 0.81| 0.64    |
| XGB_1          | 0.71     | 0.43      | 0.35   | 0.39| 0.64    |
| SGD_0          | 0.66     | 0.78      | 0.74   | 0.76| 0.62    |
| SGD_1          | 0.66     | 0.38      | 0.43   | 0.40| 0.62    |
| KNN_0          | 0.52     | 0.79      | 0.47   | 0.59| 0.58    |
| KNN_1          | 0.52     | 0.31      | 0.65   | 0.42| 0.58    |

*Insert ROC-AUC graph*

The model effectively filters out "bad" trades, though at the cost of reducing "good" trades. This trade-off is inherent, as perfect filtering is unattainable with the given market state information. However, the model stabilizes and increases the average profit per trade by over 150%, significantly improving robustness.

| Model        | ROI | # trades | AVG% | Increase, % |
|--------------|-----|----------|------|-------------|
| SVC          | 352 | 116      | 3.03 | 175         |
| RandomForest | 320 | 110      | 2.91 | 165         |
| CatBoost     | 310 | 114      | 2.72 | 147         |
| XGB          | 298 | 116      | 2.57 | 134         |
| SGD          | 269 | 161      | 1.67 | 52          |
| LogReg       | 291 | 178      | 1.63 | 48          |
| KNN          | 467 | 301      | 1.55 | 41          |

Random filtering simulations show an average profit per trade around 1%, confirming the model's effectiveness.

## Ensembling
Ensembling the top three models increased the average profit per trade to 3.5%, albeit with higher standard deviation, making it less stable but more profitable.

*Insert ROC-AUC graph*

## Tuning
Parameter tuning provided marginal performance improvements, so this step can be optional.

### Results
The final system achieved an average profit per trade of 3.5% (up from 1.1% before ML application) and an ROI of 325%, significantly outperforming the Buy-and-Hold strategy's 125% ROI. This demonstrates the system's robustness and profitability under varying market conditions.
