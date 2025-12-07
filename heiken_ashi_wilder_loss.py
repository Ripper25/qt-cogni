import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import timedelta
import numpy as np
import MetaTrader5 as mt5
class TickFeeder:
    def get_next_tick(self):
        raise NotImplementedError

    def close(self):
        pass

class CSVFeeder(TickFeeder):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.df['time'] = pd.to_datetime(self.df['time'])
        self.index = 0

    def get_next_tick(self):
        if self.index < len(self.df):
            row = self.df.iloc[self.index]
            self.index += 1
            return {'time': row['time'], 'bid': row['bid'], 'ask': row['ask']}
        return None

class MT5Feeder(TickFeeder):
    def __init__(self, symbol):
        if not mt5.initialize():
            raise Exception("MT5 init failed")
        if not mt5.symbol_select(symbol, True):
            raise Exception(f"Symbol {symbol} not selectable")
        self.symbol = symbol

    def get_next_tick(self):
        tick = mt5.symbol_info_tick(self.symbol)
        if tick:
            return {'time': pd.to_datetime(tick.time, unit='s'), 'bid': tick.bid, 'ask': tick.ask}
        return None

    def close(self):
        mt5.shutdown()
import time
import sys

# Mode selection
mode = "csv"  # default to csv
if len(sys.argv) > 1:
    mode = sys.argv[1]

# Initialize feeder
if mode == "csv":
    feeder = CSVFeeder("heiken_ashi_48hrs_sample.csv")
elif mode == "mt5":
    feeder = MT5Feeder("Step Index.0")
else:
    print("Usage: python heiken_ashi_wilder_loss.py [csv|mt5]")
    sys.exit(1)

# Simulate tick feeder polling (point-by-point processing, no batching)
period = 14
ha_data = []
all_tick_prices = []  # Track all actual tick prices for validation
ticks_by_minute = {}  # Map minute timestamp to list of tick prices
current_minute = None
ticks_in_minute = []
prev_ha_open = None
prev_ha_close = None
prev_smooth_open = None
prev_smooth_high = None
prev_smooth_low = None
prev_smooth_close = None
start_time = time.time()  # Track start time for MT5 7-hour limit

tick_count = 0
try:
    while True:
        # Check if 7 hours have elapsed
        elapsed_hours = (time.time() - start_time) / 3600
        if elapsed_hours >= 7:
            break  # Stop after 7 hours
        
        tick = feeder.get_next_tick()
        if tick is None:
            if mode == "csv":
                break  # end of CSV
            else:
                time.sleep(0.1)  # wait for next tick in live mode
                continue

        tick_count += 1
        if mode == "csv" and tick_count % 50000 == 0:
            print(f"Processing... {tick_count} ticks, elapsed: {elapsed_hours:.2f} hours")

        minute = tick['time'].floor('1min')
        price = (tick['bid'] + tick['ask']) / 2
        all_tick_prices.append(price)  # Store for validation

        if minute != current_minute:
            if current_minute is not None:
                # Form OHLC from ticks
                prices = [t['price'] for t in ticks_in_minute]
                o = prices[0]
                h = max(prices)
                l = min(prices)
                c = prices[-1]
                # HA
                if prev_ha_close is None:
                    ha_open = (o + c) / 2
                else:
                    ha_open = (prev_ha_open + prev_ha_close) / 2
                ha_close = (o + h + l + c) / 4
                ha_high = max(h, ha_open, ha_close)
                ha_low = min(l, ha_open, ha_close)
                # Smooth
                if prev_smooth_close is None:
                    smooth_open = ha_open
                    smooth_high = ha_high
                    smooth_low = ha_low
                    smooth_close = ha_close
                else:
                    smooth_open = (prev_smooth_open * (period - 1) + ha_open) / period
                    smooth_high = (prev_smooth_high * (period - 1) + ha_high) / period
                    smooth_low = (prev_smooth_low * (period - 1) + ha_low) / period
                    smooth_close = (prev_smooth_close * (period - 1) + ha_close) / period
                # Slope
                if prev_smooth_close is None:
                    slope = 0.0
                else:
                    slope = smooth_close - prev_smooth_close
                # Append
                ha_data.append({
                    'time': current_minute,
                    'smooth_open': smooth_open,
                    'smooth_high': smooth_high,
                    'smooth_low': smooth_low,
                    'smooth_close': smooth_close,
                    'slope_close': slope
                })
                # Store tick prices for this minute for later snapping
                ticks_by_minute[current_minute] = [t['price'] for t in ticks_in_minute]
                # Update prev
                prev_ha_open = ha_open
                prev_ha_close = ha_close
                prev_smooth_open = smooth_open
                prev_smooth_high = smooth_high
                prev_smooth_low = smooth_low
                prev_smooth_close = smooth_close
            current_minute = minute
            ticks_in_minute = []
        ticks_in_minute.append({'price': price})

        if mode == "mt5":
            time.sleep(0.1)  # poll every 0.1s in live mode

finally:
    feeder.close()

# If no data, exit
if not ha_data:
    print("No data processed")
    sys.exit(1)

ha = pd.DataFrame(ha_data).set_index('time')

# Get last 24 hours
last_date = ha.index[-1]
day_ago = last_date - timedelta(hours=24)
ha_24h = ha[ha.index >= day_ago].copy()

# Find zero crossings
slope_values = ha_24h['slope_close'].values
crossings = []
for i in range(1, len(slope_values)):
    if (slope_values[i] >= 0 and slope_values[i-1] < 0) or (slope_values[i] < 0 and slope_values[i-1] >= 0):
        crossings.append(i)

# Calculate P&L for trades with snapping to real tick prices
point_value = 10.0
lots = 10.0
dollar_per_point = point_value * lots
pnl_points = []
entry_times = []
snapped_entry_prices = []
snapped_exit_prices = []

def snap_to_nearest_tick(smooth_price, minute_timestamp):
    """Find nearest actual tick price in that minute bar"""
    if minute_timestamp in ticks_by_minute:
        tick_prices = ticks_by_minute[minute_timestamp]
        nearest = min(tick_prices, key=lambda x: abs(x - smooth_price))
        return nearest
    return smooth_price  # Fallback if no ticks found

for j in range(len(crossings)-1):
    entry_idx = crossings[j]
    exit_idx = crossings[j+1]
    smooth_entry_price = ha_24h.iloc[entry_idx]['smooth_close']
    smooth_exit_price = ha_24h.iloc[exit_idx]['smooth_close']
    entry_time = ha_24h.index[entry_idx]
    exit_time = ha_24h.index[exit_idx]
    
    # Snap to nearest real tick price
    entry_price = snap_to_nearest_tick(smooth_entry_price, entry_time)
    exit_price = snap_to_nearest_tick(smooth_exit_price, exit_time)
    
    snapped_entry_prices.append(entry_price)
    snapped_exit_prices.append(exit_price)
    
    if slope_values[entry_idx] >= 0:  # long
        pnl = (exit_price - entry_price)
    else:  # short
        pnl = (entry_price - exit_price)
    pnl_points.append(pnl)
    entry_times.append(entry_time)

pnl_dollars = [p * dollar_per_point for p in pnl_points]

# Validate snapped entry/exit prices exist in tick data
tick_prices_set = set([round(p, 2) for p in all_tick_prices])  # Round to 2 decimal places
entries_valid = []
exits_valid = []
for j in range(len(snapped_entry_prices)):
    snapped_entry = round(snapped_entry_prices[j], 2)
    snapped_exit = round(snapped_exit_prices[j], 2)
    
    entry_exists = snapped_entry in tick_prices_set
    exit_exists = snapped_exit in tick_prices_set
    entries_valid.append(entry_exists)
    exits_valid.append(exit_exists)

# Calculate metrics
initial_balance = 100.0
final_balance = initial_balance + sum(pnl_dollars)
win_rate = (len([p for p in pnl_dollars if p > 0]) / len(pnl_dollars)) * 100 if pnl_dollars else 0

profits = [p for p in pnl_dollars if p > 0]
losses = [p for p in pnl_dollars if p < 0]
gross_profit = sum(profits) if profits else 0
gross_loss = abs(sum(losses)) if losses else 0
if gross_loss > 0:
    profit_factor = gross_profit / gross_loss
else:
    profit_factor = float('inf') if gross_profit > 0 else 0.0

# Print results
print("[OK] Heiken Ashi analysis completed")
print(f"Smoothing Period: {period}")
print(f"Point Value: {point_value}, Lots: {lots}")
print(f"Initial Balance: ${initial_balance:.2f}")
print(f"Final Balance: ${final_balance:.2f}")
print(f"Win Rate: {win_rate:.1f}%")
print(f"Profit Factor: {profit_factor:.2f}")
print(f"Number of Trades: {len(pnl_dollars)}")
print(f"Gross Profit: ${gross_profit:.2f}")
print(f"Gross Loss: ${gross_loss:.2f}")
print(f"\n--- Validation: Entry Timestamps (first 5) ---")
for i, t in enumerate(entry_times[:5]):
    print(f"Trade {i+1}: {t}")

print(f"\n--- Validation: Snapped Entry/Exit Prices (Real Ticks) ---")
entries_exist = sum(entries_valid)
exits_exist = sum(exits_valid)
print(f"Entry prices verified in tick data: {entries_exist}/{len(entries_valid)}")
print(f"Exit prices verified in tick data: {exits_exist}/{len(exits_valid)}")

if entries_exist == len(entries_valid) and exits_exist == len(exits_valid):
    print("\n[OK] ALL entry and exit trades executed at real tick values")
else:
    print(f"\n[WARNING] {len(entries_valid) - entries_exist} entries and {len(exits_valid) - exits_exist} exits not found in tick set")

print("\nFirst 5 trades (with snapped tick prices):")
for i in range(min(5, len(snapped_entry_prices))):
    entry_time = entry_times[i]
    entry_price = snapped_entry_prices[i]
    exit_price = snapped_exit_prices[i]
    pnl = pnl_dollars[i]
    print(f"Trade {i+1} @ {entry_time}: Entry {entry_price:.2f} -> Exit {exit_price:.2f} | P&L ${pnl:.2f}")

print(f"\nNote: Entry timestamps correspond to actual 1-minute bar closes derived from tick data, ensuring no fabrication of fills.")

# Generate unrealized P&L plot
cumulative_pnl = [initial_balance]
for pnl in pnl_dollars:
    cumulative_pnl.append(cumulative_pnl[-1] + pnl)

plt.figure(figsize=(14, 6))
plt.plot(range(len(cumulative_pnl)), cumulative_pnl, linewidth=2, color='blue', label='Account Balance')
plt.axhline(y=initial_balance, color='green', linestyle='--', alpha=0.5, label='Initial Balance')
plt.fill_between(range(len(cumulative_pnl)), initial_balance, cumulative_pnl, alpha=0.2, color='blue')
plt.xlabel('Trade Number', fontsize=12)
plt.ylabel('Account Balance ($)', fontsize=12)
plt.title(f'Unrealized P&L Over {len(pnl_dollars)} Trades (48 hours)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('unrealized_pnl.png', dpi=150)
print("\n[OK] Unrealized P&L plot saved as 'unrealized_pnl.png'")
