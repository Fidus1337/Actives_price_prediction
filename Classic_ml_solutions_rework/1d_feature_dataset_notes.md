# Rebuild 1D Feature Dataset Notes

## Task Setup

- Target: classify whether `BTC close(t+1) > BTC close(t)`
- Prediction time: after the daily candle close
- Meaning: it is valid to use all daily features known at day `t`

## Goal

Build a new narrow and meaningful dataset for `1d` BTC direction prediction instead of generating a huge feature space with many weak or redundant transforms.

Recommended starting width:

- around `20-40` features total

Core idea:

- use a few strong feature families
- for each raw signal, try only `level`, `diff1`, `pct1`, and sometimes `imbalance` or short volatility
- avoid mass-generating lags and TA features at the start

## Must-Have Features

These are the first features to try in the new baseline dataset.

### 1. BTC Price Action

- `BTC close return 1d`
- `BTC close diff 1d`
- `BTC intraday range %`
- `BTC close position in candle`
- `BTC close_to_high`
- `BTC close_to_low`
- `BTC spot volume change 1d`
- `BTC realized vol 3d`
- `BTC realized vol 7d`

### 2. Open Interest / Leverage Build-Up

- `Aggregated open interest level`
- `Aggregated open interest change 1d`
- `Aggregated open interest return 1d`
- `OI / spot volume`
- `Stablecoin-margined OI share`
- `Coin-margined OI share`

### 3. Funding

- `Funding rate level`
- `Funding rate change 1d`
- `OI-weighted funding`
- `Volume-weighted funding`
- `Funding minus OI-weighted funding`

### 4. Aggressive Flow

- `Taker buy/sell imbalance`
- `Aggregated taker imbalance`
- `Taker buy volume change 1d`
- `Taker sell volume change 1d`

### 5. Forced Flow

- `Long liquidations`
- `Short liquidations`
- `Liquidation imbalance`
- `Total liquidations change 1d`

### 6. Crowding

- `Global long/short ratio`
- `Global long percent change 1d`
- `Top account long/short ratio`
- `Top account long/short change 1d`
- `Top position long/short ratio`
- `Top position long/short change 1d`

### 7. Basis / Premium

- `Coinbase premium`
- `Coinbase premium rate`
- `Coinbase premium abs`

## Good-To-Have Features

These should be tested after the must-have baseline is working.

### 1. Net Positioning

- `Net position change`
- `Net long change`
- `Net short change`
- `Cumulative net position change`

### 2. Orderbook

- `Orderbook imbalance`
- `Orderbook bids/asks change 1d`

Use only if date coverage is good and there are not too many missing periods.

### 3. Cross-Asset Regime

- `SP500 return 1d`
- `IGV return 1d`
- `Gold return 1d`

### 4. Short Regime Features

- `BTC above SMA5`
- `BTC above SMA10`
- `BTC zscore 5d`
- `BTC zscore 10d`

## External Features To Try Later

If expanding beyond CoinGlass, these are the best first candidates.

- `VIX`
- `DXY`
- `QQQ` or `NQ`
- `ETH/BTC`
- `BTC dominance`
- `TOTAL3`
- `Spot BTC ETF net flows`
- `Options ATM IV`
- `Options skew`
- `Put/call ratio`
- `Stablecoin supply change`
- `Exchange inflows/outflows`
- `Asia session return`
- `Europe session return`
- `US session return`

## What Not To Add At Start

To avoid blowing up the dataset with weak or slow signals:

- slow on-chain features like `LTH supply`, `STH supply`, `reserve risk`, `active addresses`
- large groups of TA indicators for `gold`, `sp500`, `igv`
- mass-generated `lag1`, `lag3`, `lag5`, `lag7`, `lag15` for almost every feature
- every possible `diff`, `pct`, `zscore`, and rolling transform for each raw column

## Recommended First Baseline

The first clean baseline dataset should focus only on these feature families:

- `price action`
- `open interest`
- `funding`
- `taker flow`
- `liquidations`
- `crowding`
- `Coinbase premium`

Then expand in this order:

1. `net positioning`
2. `orderbook`
3. `cross-asset regime`

## Practical Notes

- For `1d`, fast derivatives-flow features are more likely to help than slow structural on-chain features.
- A small dataset with strong signals is better than a huge dataset with hundreds of correlated features.
- It is better to compare feature families step by step than to add everything at once.
- At the start, prefer interpretability and signal quality over feature count.
