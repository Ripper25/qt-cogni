import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV data
df = pd.read_csv(r'c:\Users\Administrator\Desktop\META\Step Index_0_30days_ticks_20251023_024948.csv')

# Use bid prices for Renko calculation
prices = df['bid'].values

# Brick size is 1 point
brick_size = 1.0

# Initialize Renko bricks
closes = []
current_level = prices[0]
closes.append(current_level)

# Generate Renko bricks using all data
for price in prices[1:]:
    if price >= current_level + brick_size:
        # Up bricks
        while price >= current_level + brick_size:
            current_level += brick_size
            closes.append(current_level)
    elif price <= current_level - brick_size:
        # Down bricks
        while price <= current_level - brick_size:
            current_level -= brick_size
            closes.append(current_level)

# Use all bricks
closes = np.array(closes)

# Apply Wilder smoothing to closes (period 15, alpha=1/15)
alpha = 1.0 / 15.0
smoothed_closes = np.zeros_like(closes)
smoothed_closes[0] = closes[0]
for i in range(1, len(closes)):
    smoothed_closes[i] = smoothed_closes[i-1] + alpha * (closes[i] - smoothed_closes[i-1])

# Compute slope of smoothed closes
slopes = np.diff(smoothed_closes)

# Apply Wilder smoothing to slopes (period 15, same alpha)
smoothed_slopes = np.zeros_like(slopes)
smoothed_slopes[0] = slopes[0]
for i in range(1, len(slopes)):
    smoothed_slopes[i] = smoothed_slopes[i-1] + alpha * (slopes[i] - smoothed_slopes[i-1])

# Detect zero crossings and simulate trading
crossings = []
positions = []  # list of (entry_price, exit_price, pnl, entry_idx, exit_idx, direction)
current_position = None
current_position_price = None
entry_idx = None

for i in range(1, len(smoothed_slopes)):
    prev_sign = np.sign(smoothed_slopes[i-1])
    curr_sign = np.sign(smoothed_slopes[i])
    if prev_sign != curr_sign and prev_sign != 0 and curr_sign != 0:
        crossings.append(i)
        if curr_sign > 0:  # cross to positive, buy
            if current_position == 'sell':
                # close short position
                pnl = current_position_price - smoothed_closes[i]
                positions.append((current_position_price, smoothed_closes[i], pnl, entry_idx, i, 'sell'))
            current_position = 'buy'
            current_position_price = smoothed_closes[i]
            entry_idx = i
        elif curr_sign < 0:  # cross to negative, sell
            if current_position == 'buy':
                # close long position
                pnl = smoothed_closes[i] - current_position_price
                positions.append((current_position_price, smoothed_closes[i], pnl, entry_idx, i, 'buy'))
            current_position = 'sell'
            current_position_price = smoothed_closes[i]
            entry_idx = i

# Calculate unrealized PnL if position is open
unrealized_pnl = 0
if current_position:
    if current_position == 'buy':
        unrealized_pnl = smoothed_closes[-1] - current_position_price
    elif current_position == 'sell':
        unrealized_pnl = current_position_price - smoothed_closes[-1]

# Calculate PnL in dollars (point value 10.0, lot 0.10)
point_value = 10.0
lot_size = 0.10
positions_dollar = [(p[0], p[1], p[2] * point_value * lot_size, p[3], p[4], p[5]) for p in positions]

# Calculate adverse movements (max adverse excursion in points during each trade)
adverse_movements = []
for p in positions:
    entry_idx, exit_idx, direction = p[3], p[4], p[5]
    if direction == 'buy':
        # long position, adverse is entry_price - min(smoothed_closes during trade)
        min_during = min(smoothed_closes[entry_idx:exit_idx+1])
        adverse = p[0] - min_during
    else:
        # short position, adverse is max(smoothed_closes during trade) - entry_price
        max_during = max(smoothed_closes[entry_idx:exit_idx+1])
        adverse = max_during - p[0]
    adverse_movements.append(adverse)

# Output results
print(f"Number of zero crossings: {len(crossings)}")
total_pnl_dollar = sum(p[2] for p in positions_dollar)
print(f"Total realized PnL ($): {total_pnl_dollar}")
losses = [p for p in positions_dollar if p[2] < 0]
if losses:
    print(f"There are {len(losses)} losing trades. Example: entry {losses[0][0]}, exit {losses[0][1]}, pnl ${losses[0][2]}")
else:
    print("All trades are breakeven or profitable.")
if current_position:
    unrealized_pnl_dollar = unrealized_pnl * point_value * lot_size
    print(f"Unrealized PnL for open {current_position} position: ${unrealized_pnl_dollar}")
else:
    print("No open position.")

print(f"Adverse movements during trades (points): max {max(adverse_movements):.2f}, min {min(adverse_movements):.2f}, avg {sum(adverse_movements)/len(adverse_movements):.2f}")

# Generate account balance plot including unrealized PnL
import matplotlib.pyplot as plt
initial_balance = 10.0
balance_history = []
cumulative_realized = 0.0
for brick_idx in range(len(smoothed_closes)):
    # Add realized PnL at closing indices
    for pos in positions_dollar:
        if pos[3] == brick_idx:
            cumulative_realized += pos[2]
    current_balance = initial_balance + cumulative_realized
    if current_position and brick_idx == len(smoothed_closes) - 1:
        current_balance += unrealized_pnl_dollar
    balance_history.append(current_balance)

plt.figure(figsize=(12, 6))
plt.plot(balance_history, label='Account Balance ($)')
plt.xlabel('Brick Number')
plt.ylabel('Account Balance ($)')
plt.title('Account Balance Over Time (Starting with $10)')
plt.grid(True)
plt.legend()
plt.savefig('account_balance_plot.png')
# plt.show()  # Commented out to avoid hanging in headless mode
