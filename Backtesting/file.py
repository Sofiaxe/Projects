import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

ticker = "^OMX"
df = yf.download(ticker, start="2015-01-01", end="2025-01-01",
                 auto_adjust=True, progress=False)
df.columns = df.columns.get_level_values(0)

# Moving averages
df['SMA_50'] = df['Close'].rolling(50).mean()
df['SMA_200'] = df['Close'].rolling(200).mean()

# Buy and hold
total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
print(f"Buy and Hold Return: {total_return:.2f}%")

# Signals (integer, not boolean!)
df['Signal'] = (df['SMA_50'] > df['SMA_200']).astype(int)
df['Position'] = df['Signal'].diff()

bt = df.dropna(subset=['SMA_200']).copy()

bt['Return'] = bt['Close'].pct_change().fillna(0)
bt['Strategy_Return'] = bt['Return']*bt['Signal'].shift(1).fillna(0)

bt['BuyHold'] = (1 + bt['Return']).cumprod()
bt['Strategy'] = (1 + bt['Strategy_Return']).cumprod()

def stats(curve, rets):
    years = (curve.index[-1] - curve.index[0]).days / 365.25
    total = (curve.iloc[-1] - 1) * 100
    cagr = (curve.iloc[-1] ** (1 / years) - 1) * 100
    max_dd = ((curve / curve.cummax()) - 1).min() * 100
    sharpe = rets.mean() / rets.std() * (252 ** 0.5)
    return f"Total {total:7.1f}% | CAGR {cagr:5.1f}% | MaxDD {max_dd:6.1f}% | Sharpe {sharpe:4.2f}"

print("Buy & Hold:", stats(bt['BuyHold'], bt['Return']))
print("SMA Cross: ", stats(bt['Strategy'], bt['Strategy_Return']))
print("Number of buys: ", (bt['Position'] == 1).sum())


bt[['BuyHold', 'Strategy']].plot(figsize=(12, 6), grid=True,
                                 title=f'{ticker}: SMA 50/200 vs Buy & Hold')
# Plot
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], label='Close Price', alpha=0.7)
plt.plot(df.index, df['SMA_50'], label='50-Day SMA', color='green', linewidth=2)
plt.plot(df.index, df['SMA_200'], label='200-Day SMA', color='red', linewidth=2)

buys = df[df['Position'] == 1]
sells = df[df['Position'] == -1]
plt.plot(buys.index, buys['SMA_50'], '^', markersize=10, color='g', label='Buy signal')
plt.plot(sells.index, sells['SMA_50'], 'v', markersize=10, color='r', label='Sell signal')

plt.title(f'{ticker} Close Price, 50 and 200 day SMA')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()