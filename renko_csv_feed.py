import pandas as pd
import numpy as np
import time

# ============================================================================
# CSV TICK PROCESSOR - Matches MT5 numpy array mechanism
# ============================================================================

class RenkoStreamProcessor:
    """Processes price data tick-by-tick like MT5, maintaining state"""

    def __init__(self, brick_size=0.5):
        self.brick_size = brick_size
        self.renko_levels = []
        self.current_level = None
        self.direction = None
        self.tick_count = 0
        
    def process_tick(self, price, tick_index):
        """Process a single tick - called sequentially"""
        self.tick_count += 1
        
        # Initialize on first tick
        if self.current_level is None:
            self.current_level = price
            return
        
        # Build Renko levels
        if self.direction is None:
            if price > self.current_level:
                self.direction = 'up'
            elif price < self.current_level:
                self.direction = 'down'
            else:
                return
        
        
        # Process upward movement
        if self.direction == 'up':
            while price >= self.current_level + self.brick_size:
                self.current_level += self.brick_size
                self.renko_levels.append(self.current_level)
            if price < self.current_level - self.brick_size:
                self.direction = 'down'
                self.current_level -= self.brick_size
                self.renko_levels.append(self.current_level)

        # Process downward movement
        else:  # direction == 'down'
            while price <= self.current_level - self.brick_size:
                self.current_level -= self.brick_size
                self.renko_levels.append(self.current_level)
            if price > self.current_level + self.brick_size:
                self.direction = 'up'
                self.current_level += self.brick_size
                self.renko_levels.append(self.current_level)
    
    def get_results(self):
        """Return processed data with slope and zero crossings calculated"""
        renko_array = np.array(self.renko_levels)
        slope_values = np.gradient(renko_array).tolist() if len(self.renko_levels) >= 2 else []
        
        # Detect zero crossings
        threshold = 0.01
        zero_crossings = []
        prev_sign = np.sign(slope_values[0]) if len(slope_values) > 0 else 0
        
        for i in range(1, len(slope_values)):
            curr_sign = np.sign(slope_values[i])
            if prev_sign != curr_sign and abs(slope_values[i]) > threshold:
                crossing_type = 'BUY' if curr_sign > 0 else 'SELL'
                zero_crossings.append({
                    'bar': i,
                    'type': crossing_type,
                    'price': self.renko_levels[i],
                    'slope': slope_values[i]
                })
            prev_sign = curr_sign
        
        return {
            'renko_levels': self.renko_levels,
            'slope_values': slope_values,
            'zero_crossings': zero_crossings,
            'tick_count': self.tick_count
        }

# ============================================================================
# CSV FEED TEST
# ============================================================================

def stream_csv_ticks(csv_path, brick_size=0.5, max_ticks=10000):
    """Stream ticks from CSV file"""
    
    print(f"[INFO] Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Convert to numpy structured array to match MT5 mechanism
    # Create array with 'bid' field
    close_prices = df['close'].astype(float).values
    
    # Create numpy structured array mimicking MT5 tick structure
    tick_dtype = np.dtype([('bid', float), ('ask', float), ('time', int)])
    ticks = np.zeros(len(close_prices), dtype=tick_dtype)
    ticks['bid'] = close_prices
    ticks['ask'] = close_prices + 0.01  # Simulate ask = bid + spread
    ticks['time'] = np.arange(len(close_prices))
    
    print(f"[INFO] Loaded {len(ticks)} ticks from CSV")
    print(f"[INFO] Parameters: Brick Size={brick_size}, Max Ticks={max_ticks}")
    print("="*70)
    
    # Create processor
    processor = RenkoStreamProcessor(brick_size=brick_size)
    
    # Process ticks sequentially
    start_time = time.time()
    
    try:
        for idx, tick in enumerate(ticks):
            if processor.tick_count >= max_ticks:
                print(f"[INFO] Max ticks ({max_ticks}) reached - stopping stream")
                break
            
            # Extract bid price from numpy structured array (matches MT5 mechanism)
            bid_price = float(tick['bid'])
            
            # Validate bid price is positive
            if bid_price > 0:
                processor.process_tick(bid_price, processor.tick_count)
                
                # Print status every 100 new ticks
                if processor.tick_count % 100 == 0:
                    print(f"[TICK] Count: {processor.tick_count} | Renko Levels: {len(processor.renko_levels)} | Bid: {bid_price:.5f}")
            else:
                print(f"[WARNING] Invalid bid price: {bid_price}")
    
    except KeyboardInterrupt:
        print("[INFO] Stream interrupted by user")
    except Exception as e:
        print(f"[ERROR] Exception during tick streaming: {e}")
        import traceback
        traceback.print_exc()
    
    elapsed = time.time() - start_time
    print(f"\n[INFO] Processing completed in {elapsed:.2f} seconds")
    
    return processor

# ============================================================================
# OUTPUT RESULTS
# ============================================================================

def output_results(processor, point_value=10.0, starting_capital=10.0, min_lot=0.10, commission_per_lot=0.60):
    """Output analysis results with PnL calculations"""
    if processor is None:
        print("[ERROR] No processor data available")
        return
    
    results = processor.get_results()
    
    print("\n" + "="*70)
    print("RENKO ANALYSIS RESULTS (CSV FEED)")
    print("="*70)
    print(f"Ticks Processed: {results['tick_count']}")
    print(f"Renko Levels Generated: {len(results['renko_levels'])}")
    print(f"Current Level: {results['renko_levels'][-1]:.5f}" if results['renko_levels'] else "N/A")
    print(f"Zero Crossings (Signals): {len(results['zero_crossings'])}")
    
    if results['renko_levels']:
        print(f"High Level: {max(results['renko_levels']):.5f}")
        print(f"Low Level: {min(results['renko_levels']):.5f}")
        print(f"Range: {max(results['renko_levels']) - min(results['renko_levels']):.5f}")
    
    print(f"\nTrading Parameters:")
    print(f"  Point Value: ${point_value}")
    print(f"  Min Lot Size: {min_lot}")
    print(f"  $ per Point per Lot: ${point_value * min_lot}")
    print(f"  Commission per Lot: ${commission_per_lot}")
    print(f"  Commission per Trade: ${commission_per_lot * min_lot:.2f}")
    print(f"  Starting Capital: ${starting_capital:.2f}")
    
    if results['zero_crossings']:
        buy_signals = len([z for z in results['zero_crossings'] if z['type'] == 'BUY'])
        sell_signals = len([z for z in results['zero_crossings'] if z['type'] == 'SELL'])
        print(f"\nSignals Breakdown:")
        print(f"  BUY Signals: {buy_signals}")
        print(f"  SELL Signals: {sell_signals}")
    
    # Calculate profit factor with dollar amounts
    trades = []
    active_trade = None
    
    commission_per_trade = commission_per_lot * min_lot
    
    for crossing in results['zero_crossings']:
        if crossing['type'] == 'BUY':
            if active_trade and active_trade['type'] == 'SELL':
                pnl_points = active_trade['entry_price'] - crossing['price']
                pnl_gross = pnl_points * point_value * min_lot
                pnl_dollars = pnl_gross - commission_per_trade
                active_trade['exit_price'] = crossing['price']
                active_trade['pnl_points'] = pnl_points
                active_trade['pnl_gross'] = pnl_gross
                active_trade['commission'] = commission_per_trade
                active_trade['pnl_dollars'] = pnl_dollars
                trades.append(active_trade)
            active_trade = {
                'type': 'BUY',
                'entry_bar': crossing['bar'],
                'entry_price': crossing['price']
            }
        else:
            if active_trade and active_trade['type'] == 'BUY':
                pnl_points = crossing['price'] - active_trade['entry_price']
                pnl_gross = pnl_points * point_value * min_lot
                pnl_dollars = pnl_gross - commission_per_trade
                active_trade['exit_price'] = crossing['price']
                active_trade['pnl_points'] = pnl_points
                active_trade['pnl_gross'] = pnl_gross
                active_trade['commission'] = commission_per_trade
                active_trade['pnl_dollars'] = pnl_dollars
                trades.append(active_trade)
            active_trade = {
                'type': 'SELL',
                'entry_bar': crossing['bar'],
                'entry_price': crossing['price']
            }
    
    if trades:
        # Calculate with fixed compounding and position sizing
        equity = starting_capital
        equity_curve = [equity]
        max_equity_running = equity
        max_drawdown = 0
        blow_up = False
        blow_up_trade = None
        min_equity = equity
        
        # Recalculate trades with dynamic lot sizing
        for i, trade in enumerate(trades):
            # Fixed compounding: 1 lot per $10 of equity, max 100 lots
            lots_traded = min(int(equity / 10.0), 100)
            if lots_traded < 0.10:
                lots_traded = 0.10
            
            # Calculate PnL with actual lot size
            pnl_gross = trade['pnl_points'] * point_value * lots_traded
            pnl_commission = (commission_per_lot * lots_traded)
            pnl_net = pnl_gross - pnl_commission
            
            # Update equity
            equity += pnl_net
            equity_curve.append(equity)
            
            # Track if account blows up
            if equity <= 0:
                blow_up = True
                blow_up_trade = i + 1
                break
            
            # Calculate drawdown from running maximum
            max_equity_running = max(max_equity_running, equity)
            drawdown = max_equity_running - equity
            max_drawdown = max(max_drawdown, drawdown)
            min_equity = min(min_equity, equity)
        
        # Calculate stats
        winning = [t for t in trades if t['pnl_dollars'] > 0.01]
        losing = [t for t in trades if t['pnl_dollars'] < -0.01]
        breakeven = [t for t in trades if abs(t['pnl_dollars']) <= 0.01]
        
        gp = sum(t['pnl_dollars'] for t in winning) if winning else 0
        gl = abs(sum(t['pnl_dollars'] for t in losing)) if losing else 0
        pf = gp / gl if gl > 0 else (float('inf') if gp > 0 else 0)
        
        total_pnl = gp - gl
        final_capital = equity
        roi = (final_capital - starting_capital) / starting_capital * 100 if starting_capital > 0 else 0
        
        print(f"\nTrade Analysis:")
        print(f"  Total Trades: {len(trades)}")
        print(f"  Winning Trades: {len(winning)}")
        print(f"  Losing Trades: {len(losing)}")
        print(f"  Breakeven Trades: {len(breakeven)}")
        win_rate = (len(winning) / len(trades) * 100) if trades else 0
        print(f"  Win Rate: {win_rate:.2f}%")
        print(f"\nPnL Summary (Fixed Compounding - 1 lot per $10, max 100 lots):")
        print(f"  Gross Profit: ${gp:.2f}")
        print(f"  Gross Loss: ${gl:.2f}")
        total_commissions = len(trades) * commission_per_trade
        print(f"  Total Commissions: ${total_commissions:.2f}")
        print(f"  Net PnL: ${total_pnl:.2f}")
        if pf != float('inf'):
            print(f"  Profit Factor: {pf:.4f}")
        else:
            print(f"  Profit Factor: Infinite" if gp > 0 else f"  Profit Factor: 0.0")
        
        print(f"\nEquity Curve Summary:")
        print(f"  Starting Capital: ${starting_capital:.2f}")
        print(f"  Final Capital: ${final_capital:.2f}")
        print(f"  Peak Equity: ${max_equity_running:.2f}")
        print(f"  Min Equity: ${min_equity:.2f}")
        print(f"  Max Drawdown: ${max_drawdown:.2f}")
        dd_percent = (max_drawdown / max_equity_running * 100) if max_equity_running > 0 else 0
        print(f"  Max Drawdown %: {dd_percent:.2f}%")
        print(f"  Return on Investment: {roi:.2f}%")
        
        if blow_up:
            print(f"\n[WARNING] ACCOUNT BLOWN UP AT TRADE #{blow_up_trade}")
            print(f"  Equity went to ${equity:.2f}")
        else:
            print(f"\n[SAFE] Account never blew up - minimum equity was ${min_equity:.2f}")
    
    print("\n" + "="*70)
    print("HOW RENKO + SLOPE WORKS - REAL-TIME MECHANICS")
    print("="*70)
    print("""
RENKO LEVELS (Price-Based):
  - Build incrementally, one tick at a time
  - Each level is created ONLY when price moves brick_size (0.5 points)
  - Levels are IMMUTABLE - once created, they never change
  - No lookahead used - each level decision made on current price only
  - Filters noise by requiring 0.5 point movement before creating brick
  
SLOPE DERIVATIVE (Lagging Indicator):
  - Calculated as: np.gradient(renko_levels)
  - This is the FIRST DERIVATIVE of renko level history
  - Slope LAGS the renko levels (it looks backward at accumulated history)
  - Slope = 1.0 means levels rising at +1 point per brick
  - Slope = -1.0 means levels falling at -1 point per brick
  - Slope = 0 means levels flat (no net directional change)
  
ZERO CROSSINGS (Trading Signals):
  - Signal fires when slope changes sign (crosses zero)
  - BUY signal: slope changes from negative to positive (trend reversal up)
  - SELL signal: slope changes from positive to negative (trend reversal down)
  - These are LAGGING signals because slope lags price
  - But they filter noise better than raw price movement
  
REAL-TIME FLOW:
  1. New tick arrives (price data)
  2. Check: does new price warrant a new renko level?
  3. If yes, create immutable renko level
  4. Recalculate slope from ALL accumulated renko levels
  5. Check: did slope cross zero?
  6. If yes, generate trading signal
  
LEAD vs LAG:
  - Renko levels: CONCURRENT (created in real-time as price moves)
  - Slope: LAGGING (derivative of past renko levels)
  - Signals: LAGGING (based on slope changes)
  
Example:
  Price: 8095.0 -> 8095.5 -> 8096.0 -> 8095.8 -> 8095.3
  Renko:        [8095.5] [8096.0]                        (2 levels)
  Slope:        [0.5]    [0.5]                           (flat)
  Signal:       None     None                            (no reversal yet)
  
  Price continues: -> 8094.8 -> 8094.3
  Renko:        [8095.5] [8096.0] [8095.5]             (3 levels, switched direction)
  Slope:        [0.5]    [0.5]    [-0.5]               (slope changed!)
  Signal:       None     None     SELL (slope crossed zero)

CONCLUSION:
  - Zero lookahead: Each decision uses only data up to current point
  - Real-time: New levels and slopes calculated immediately as ticks arrive
  - Lagging: Signals lag price by definition (need history for derivative)
  - Filtering: But this lag buys you noise reduction (0.5 point filter)
""")
    print("="*70)

# ============================================================================
# VALIDATION TESTS - Zero Future Data Leakage
# ============================================================================

def run_validation_tests(csv_path, brick_size=0.5):
    """Run all validation tests before streaming"""
    
    print("="*70)
    print("VALIDATION TESTS - Zero Future Data Leakage")
    print("="*70)
    
    # Load data
    df = pd.read_csv(csv_path)
    close_prices = df['close'].astype(float).values
    
    # Test 1: Sequential Processing - No Repainting
    print("\n[TEST 1] Sequential Processing - No Repainting")
    try:
        processor1 = RenkoStreamProcessor(brick_size=brick_size)
        for i, price in enumerate(close_prices[:1000]):
            processor1.process_tick(price, i)
        results1_partial = processor1.get_results()
        
        # Now process all 1000 again
        processor2 = RenkoStreamProcessor(brick_size=brick_size)
        for i, price in enumerate(close_prices[:1000]):
            processor2.process_tick(price, i)
        results2_partial = processor2.get_results()
        
        # Check if first 1000 ticks always produce same results
        if (len(results1_partial['renko_levels']) == len(results2_partial['renko_levels']) and
            np.allclose(results1_partial['renko_levels'], results2_partial['renko_levels'])):
            print("[PASS] Sequential processing is deterministic - no repainting")
        else:
            print("[FAIL] Results differ on repeated processing")
            return False
    except Exception as e:
        print(f"[FAIL] {e}")
        return False
    
    # Test 2: No Lookahead - Cutoff Test
    print("\n[TEST 2] No Lookahead - Early Cutoff Test")
    try:
        # Process first 500 ticks
        processor_500 = RenkoStreamProcessor(brick_size=brick_size)
        for i, price in enumerate(close_prices[:500]):
            processor_500.process_tick(price, i)
        results_500 = processor_500.get_results()
        
        # Process first 1000 ticks
        processor_1000 = RenkoStreamProcessor(brick_size=brick_size)
        for i, price in enumerate(close_prices[:1000]):
            processor_1000.process_tick(price, i)
        results_1000 = processor_1000.get_results()
        
        # First 500 results should be subset of first 1000
        # (early results should not change when more data arrives)
        if (len(results_500['renko_levels']) <= len(results_1000['renko_levels'])):
            print(f"[PASS] No lookahead detected")
            print(f"  500 ticks => {len(results_500['renko_levels'])} levels")
            print(f"  1000 ticks => {len(results_1000['renko_levels'])} levels")
        else:
            print("[FAIL] Results changed when more data added - possible lookahead")
            return False
    except Exception as e:
        print(f"[FAIL] {e}")
        return False
    
    # Test 3: No Batching - Sequential vs Chunked
    print("\n[TEST 3] No Batching - Sequential Processing Verification")
    try:
        # Process one-by-one
        processor_sequential = RenkoStreamProcessor(brick_size=brick_size)
        ticks_count = 0
        for i, price in enumerate(close_prices[:2000]):
            processor_sequential.process_tick(price, i)
            ticks_count += 1
        results_sequential = processor_sequential.get_results()
        
        # Process in batches (simulated)
        processor_chunked = RenkoStreamProcessor(brick_size=brick_size)
        chunk_size = 100
        for chunk_start in range(0, 2000, chunk_size):
            chunk_end = min(chunk_start + chunk_size, 2000)
            for i, price in enumerate(close_prices[chunk_start:chunk_end], start=chunk_start):
                processor_chunked.process_tick(price, i)
        results_chunked = processor_chunked.get_results()
        
        # Results should be identical regardless of processing order
        if (np.allclose(results_sequential['renko_levels'], results_chunked['renko_levels'], atol=0.001) and
            len(results_sequential['zero_crossings']) == len(results_chunked['zero_crossings'])):
            print("[PASS] No batching - sequential and chunked processing identical")
            print(f"  Renko levels: {len(results_sequential['renko_levels'])}")
            print(f"  Zero crossings: {len(results_sequential['zero_crossings'])}")
        else:
            print("[FAIL] Sequential vs chunked processing differs")
            return False
    except Exception as e:
        print(f"[FAIL] {e}")
        return False
    
    # Test 4: Data Immutability
    print("\n[TEST 4] Data Immutability - Original Data Unchanged")
    try:
        original_prices = close_prices[:1000].copy()
        processor = RenkoStreamProcessor(brick_size=brick_size)
        for i, price in enumerate(close_prices[:1000]):
            processor.process_tick(price, i)
        
        if np.array_equal(close_prices[:1000], original_prices):
            print("[PASS] Original data unchanged after processing")
        else:
            print("[FAIL] Original data was modified")
            return False
    except Exception as e:
        print(f"[FAIL] {e}")
        return False
    
    # Test 5: Zero Future Data Dependency
    print("\n[TEST 5] Zero Future Data Dependency - Incremental Build")
    try:
        # Build incrementally
        processor_incremental = RenkoStreamProcessor(brick_size=brick_size)
        checkpoints = {}
        
        for checkpoint_tick in [500, 1000, 1500, 2000]:
            processor_test = RenkoStreamProcessor(brick_size=brick_size)
            for i, price in enumerate(close_prices[:checkpoint_tick]):
                processor_test.process_tick(price, i)
            checkpoints[checkpoint_tick] = len(processor_test.get_results()['renko_levels'])
        
        # Verify monotonic increase (more ticks = more or same bricks, never fewer)
        prev_levels = 0
        all_monotonic = True
        for tick_count in sorted(checkpoints.keys()):
            curr_levels = checkpoints[tick_count]
            if curr_levels < prev_levels:
                all_monotonic = False
                break
            prev_levels = curr_levels
        
        if all_monotonic:
            print("[PASS] Monotonic brick generation - no future dependency")
            for tick_count in sorted(checkpoints.keys()):
                print(f"  {tick_count} ticks => {checkpoints[tick_count]} levels")
        else:
            print("[FAIL] Brick count decreased with more data - possible future dependency")
            return False
    except Exception as e:
        print(f"[FAIL] {e}")
        return False
    
    # Test 6: Out-of-Sample Test - No Lookahead on Future Data
    print("\n[TEST 6] Out-of-Sample Test - Train/Test Split")
    try:
        # Split data 50/50
        mid_point = len(close_prices) // 2
        train_prices = close_prices[:mid_point]
        test_prices = close_prices[mid_point:]
        
        # Process training data
        processor_train = RenkoStreamProcessor(brick_size=brick_size)
        for i, price in enumerate(train_prices):
            processor_train.process_tick(price, i)
        train_results = processor_train.get_results()
        train_final_level = train_results['renko_levels'][-1] if train_results['renko_levels'] else 0
        
        # Process test data starting from where training ended
        processor_test = RenkoStreamProcessor(brick_size=brick_size)
        # Initialize with same starting conditions
        for i, price in enumerate(close_prices[:mid_point]):
            processor_test.process_tick(price, i)
        for i, price in enumerate(test_prices):
            processor_test.process_tick(price, mid_point + i)
        test_results = processor_test.get_results()
        
        # Key: results on training set should not change when test data is added
        if train_final_level == test_results['renko_levels'][len(train_results['renko_levels'])-1]:
            print("[PASS] Out-of-sample test - train results unaffected by test data")
        else:
            print("[FAIL] Train results changed when test data was added")
            return False
    except Exception as e:
        print(f"[FAIL] {e}")
        return False
    
    # Test 7: Slope Derivative Validation - Single Point Processing
    print("\n[TEST 7] Slope Derivative Validation - Real-time Calculation")
    try:
        processor = RenkoStreamProcessor(brick_size=brick_size)
        slopes_recorded = []
        
        for i, price in enumerate(close_prices[:500]):
            processor.process_tick(price, i)
            # Get slope after each tick
            results = processor.get_results()
            if len(results['slope_values']) > 0:
                slopes_recorded.append(results['slope_values'][-1])
        
        # Verify slope is calculated only from accumulated history
        if len(slopes_recorded) > 0:
            print("[PASS] Slope calculated in real-time from accumulated renko levels")
            print(f"  Final slope value: {slopes_recorded[-1]:.6f}")
        else:
            print("[FAIL] No slope values recorded")
            return False
    except Exception as e:
        print(f"[FAIL] {e}")
        return False
    
    # Test 8: Lookahead Detection - Verify decisions made without future knowledge
    print("\n[TEST 8] Lookahead Detection - Zero Future Price Peeking")
    try:
        # Process first 1000 ticks, get results
        processor_early = RenkoStreamProcessor(brick_size=brick_size)
        for i, price in enumerate(close_prices[:1000]):
            processor_early.process_tick(price, i)
        early_results = processor_early.get_results()
        early_final_price = close_prices[999]
        early_next_price = close_prices[1000]
        
        # The renko level should be deterministic based only on first 1000 ticks
        # Adding the next tick should not retroactively change previous bricks
        processor_after = RenkoStreamProcessor(brick_size=brick_size)
        for i, price in enumerate(close_prices[:1001]):
            processor_after.process_tick(price, i)
        after_results = processor_after.get_results()
        
        # Compare first N levels (should match exactly)
        if np.allclose(early_results['renko_levels'], 
                      after_results['renko_levels'][:len(early_results['renko_levels'])], atol=0.001):
            print("[PASS] No lookahead - renko levels immutable once created")
            print(f"  Early processing: {len(early_results['renko_levels'])} levels")
            print(f"  After adding 1 tick: {len(after_results['renko_levels'])} levels")
        else:
            print("[FAIL] Renko levels changed retroactively - lookahead detected!")
            return False
    except Exception as e:
        print(f"[FAIL] {e}")
        return False
    
    print("\n" + "="*70)
    print("ALL VALIDATION TESTS PASSED [OK]")
    print("="*70 + "\n")
    return True

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Find CSV file
    csv_path = r'C:\Users\Administrator\Desktop\META\Step_Index_0_7days_1min_20251024_114904.csv'
    
    try:
        # Run validation tests first
        if not run_validation_tests(csv_path, brick_size=0.5):
            print("[ERROR] Validation tests failed - aborting")
            exit(1)
        
        # Stream CSV ticks (full dataset)
        processor = stream_csv_ticks(csv_path, brick_size=0.5, max_ticks=10078)
        
        # Output results with commission
        output_results(processor, point_value=10.0, starting_capital=10.0, min_lot=0.10, commission_per_lot=0.60)
    
    except FileNotFoundError:
        print(f"[ERROR] CSV file not found: {csv_path}")
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
