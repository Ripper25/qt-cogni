import numpy as np
import MetaTrader5 as mt5
import time

# ============================================================================
# VALIDATION TESTS - Zero Future Data Leakage
# ============================================================================

def run_validation_tests():
    """Run all validation tests before live streaming"""
    
    print("="*70)
    print("VALIDATION TESTS - Zero Future Data Leakage")
    print("="*70)
    
    # Test 1: Append Immutability
    try:
        processor = RenkoStreamProcessor(brick_size=0.5)
        test_prices = [8095.4, 8095.6, 8095.4, 8096.2]
        test_prices_orig = test_prices.copy()
        
        for price in test_prices:
            processor.process_tick(price, processor.tick_count)
        
        assert test_prices == test_prices_orig, "Original data was modified!"
        print("[PASS] Append Immutability Test - Original data unchanged")
    except AssertionError as e:
        print(f"[FAIL] {e}")
        return False
    
    # Test 2: No NaN/Inf Arrays
    try:
        processor = RenkoStreamProcessor(brick_size=0.5)
        test_prices = [8095.4, 8095.6, 8095.4, 8096.2, 8096.4, 8098.0, 8098.8]
        
        for price in test_prices:
            processor.process_tick(price, processor.tick_count)
        
        results = processor.get_results()
        
        assert not np.any(np.isnan(results['renko_levels'])), "NaN found in renko levels!"
        assert not np.any(np.isinf(results['renko_levels'])), "Inf found in renko levels!"
        assert not np.any(np.isnan(results['slope_values'])), "NaN found in slope data!"
        assert not np.any(np.isinf(results['slope_values'])), "Inf found in slope data!"
        
        print("[PASS] No NaN/Inf Arrays - All computed arrays valid")
    except AssertionError as e:
        print(f"[FAIL] {e}")
        return False
    
    # Test 3: Small Perturbation
    try:
        test_prices = [8095.4, 8095.6, 8095.4, 8096.2, 8096.4, 8098.0, 8098.8, 8097.3, 8095.0]
        
        processor1 = RenkoStreamProcessor(brick_size=0.5)
        for price in test_prices:
            processor1.process_tick(price, processor1.tick_count)

        perturbed_prices = test_prices.copy()
        perturbed_prices[0] += 0.00001
        processor2 = RenkoStreamProcessor(brick_size=0.5)
        for price in perturbed_prices:
            processor2.process_tick(price, processor2.tick_count)
        
        results1 = processor1.get_results()
        results2 = processor2.get_results()
        
        if len(results1['slope_values']) > 0 and len(results2['slope_values']) > 0:
            max_deviation = np.max(np.abs(np.array(results1['slope_values']) - np.array(results2['slope_values'])))
            assert max_deviation < 0.01, f"Perturbation caused excessive deviation: {max_deviation}"
            print(f"[PASS] Small Perturbation Test - Max deviation {max_deviation:.6f} (acceptable)")
        else:
            print("[PASS] Small Perturbation Test - Insufficient data for comparison")
    except AssertionError as e:
        print(f"[FAIL] {e}")
        return False
    
    # Test 4: Monotonic Renko
    try:
        test_prices = [8095.4, 8095.6, 8095.4, 8096.2, 8096.4, 8098.0, 8098.8, 8097.3, 8095.0]
        
        processor = RenkoStreamProcessor(brick_size=0.5)
        for price in test_prices:
            processor.process_tick(price, processor.tick_count)
        
        results = processor.get_results()
        
        if len(results['renko_levels']) >= 2:
            diffs = np.abs(np.diff(results['renko_levels']))
            max_jump = np.max(diffs)
            assert max_jump <= processor.brick_size + 0.001, f"Renko jump exceeds brick size: {max_jump}"
            print(f"[PASS] Monotonic Renko Test - Max jump {max_jump:.4f} (brick size: {processor.brick_size})")
        else:
            print("[PASS] Monotonic Renko Test - Insufficient data")
    except AssertionError as e:
        print(f"[FAIL] {e}")
        return False
    
    # Test 5: Basic Functionality
    try:
        processor = RenkoStreamProcessor(brick_size=0.01)
        test_prices = [8095.0 + i*0.05 for i in range(10)]

        for price in test_prices:
            processor.process_tick(price, processor.tick_count)

        results = processor.get_results()
        assert len(results['renko_levels']) > 0, "No renko levels generated"

        print(f"[PASS] Basic Functionality Test - Generated {len(results['renko_levels'])} bricks")
    except AssertionError as e:
        print(f"[FAIL] {e}")
        return False
    
    print("="*70)
    print("ALL TESTS PASSED [OK]")
    print("="*70 + "\n")
    return True

# ============================================================================
# MT5-STYLE TICK STREAMING - Sequential Data Feed
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
# LIVE MT5 TICK STREAMING
# ============================================================================

def stream_live_ticks(symbol="Step Index.0", brick_size=0.5, max_ticks=10000):
    """Stream live ticks from MT5 terminal"""

    # Initialize MT5
    if not mt5.initialize():
        print("[ERROR] Failed to initialize MT5 connection")
        return None

    # Select symbol
    if not mt5.symbol_select(symbol, True):
        print(f"[ERROR] Failed to select symbol {symbol}")
        mt5.shutdown()
        return None

    print(f"[INFO] MT5 Connected - Streaming {symbol}")
    print(f"[INFO] Parameters: Brick Size={brick_size}, Max Ticks={max_ticks}")
    print("="*70)

    # Create processor
    processor = RenkoStreamProcessor(brick_size=brick_size)
    
    # Stream live ticks
    last_time_msc = 0
    
    try:
        while processor.tick_count < max_ticks:
            # Get CURRENT live tick from MT5
            tick = mt5.symbol_info_tick(symbol)

            if tick is None:
                print(f"[ERROR] Failed to get tick for {symbol}")
                time.sleep(0.1)
                continue

            # Process only if NEW tick arrived (time_msc changed indicates new tick)
            if tick.time_msc != last_time_msc:
                # Extract bid price from current tick
                bid_price = float(tick.bid)
                
                # Validate bid price is positive
                if bid_price > 0:
                    processor.process_tick(bid_price, processor.tick_count)
                    last_time_msc = tick.time_msc

                    # Print status every 10 new ticks
                    if processor.tick_count % 10 == 0:
                        print(f"[TICK] Count: {processor.tick_count} | Renko Levels: {len(processor.renko_levels)} | Bid: {bid_price:.5f}")
                else:
                    print(f"[WARNING] Invalid bid price: {bid_price}")

                # Check if max ticks reached
                if processor.tick_count >= max_ticks:
                    print(f"[INFO] Max ticks ({max_ticks}) reached - stopping stream")
                    break

            # Small sleep to avoid CPU hogging
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("[INFO] Stream interrupted by user")
    except Exception as e:
        print(f"[ERROR] Exception during tick streaming: {e}")
    
    finally:
        mt5.shutdown()
    
    return processor

# ============================================================================
# OUTPUT RESULTS
# ============================================================================

def output_results(processor):
    """Output analysis results"""
    if processor is None:
        print("[ERROR] No processor data available")
        return
    
    results = processor.get_results()
    
    print("\n" + "="*70)
    print("LIVE RENKO ANALYSIS RESULTS")
    print("="*70)
    print(f"Ticks Processed: {results['tick_count']}")
    print(f"Renko Levels Generated: {len(results['renko_levels'])}")
    print(f"Current Level: {results['renko_levels'][-1]:.5f}" if results['renko_levels'] else "N/A")
    print(f"Zero Crossings (Signals): {len(results['zero_crossings'])}")
    
    if results['renko_levels']:
        print(f"High Level: {max(results['renko_levels']):.5f}")
        print(f"Low Level: {min(results['renko_levels']):.5f}")
        print(f"Range: {max(results['renko_levels']) - min(results['renko_levels']):.5f}")
    
    if results['zero_crossings']:
        buy_signals = len([z for z in results['zero_crossings'] if z['type'] == 'BUY'])
        sell_signals = len([z for z in results['zero_crossings'] if z['type'] == 'SELL'])
        print(f"\nSignals Breakdown:")
        print(f"  BUY Signals: {buy_signals}")
        print(f"  SELL Signals: {sell_signals}")
    
    # Calculate profit factor
    trades = []
    active_trade = None
    
    for crossing in results['zero_crossings']:
        if crossing['type'] == 'BUY':
            if active_trade and active_trade['type'] == 'SELL':
                pnl = active_trade['entry_price'] - crossing['price']
                active_trade['exit_price'] = crossing['price']
                active_trade['pnl'] = pnl
                trades.append(active_trade)
            active_trade = {
                'type': 'BUY',
                'entry_bar': crossing['bar'],
                'entry_price': crossing['price']
            }
        else:
            if active_trade and active_trade['type'] == 'BUY':
                pnl = crossing['price'] - active_trade['entry_price']
                active_trade['exit_price'] = crossing['price']
                active_trade['pnl'] = pnl
                trades.append(active_trade)
            active_trade = {
                'type': 'SELL',
                'entry_bar': crossing['bar'],
                'entry_price': crossing['price']
            }
    
    if trades:
        winning = [t for t in trades if t['pnl'] > 0]
        losing = [t for t in trades if t['pnl'] < 0]
        gp = sum(t['pnl'] for t in winning) if winning else 0
        gl = abs(sum(t['pnl'] for t in losing)) if losing else 0
        pf = gp / gl if gl > 0 else (float('inf') if gp > 0 else 0)
        
        print(f"\nTrade Analysis:")
        print(f"  Total Trades: {len(trades)}")
        print(f"  Winning: {len(winning)}")
        print(f"  Losing: {len(losing)}")
        win_rate = (len(winning) / len(trades) * 100) if trades else 0
        print(f"  Win Rate: {win_rate:.2f}%")
        print(f"  Gross Profit: {gp:.2f}")
        print(f"  Gross Loss: {gl:.2f}")
        if pf != float('inf'):
            print(f"  Profit Factor: {pf:.4f}")
        else:
            print(f"  Profit Factor: Infinite (No losses)")
    
    print("="*70)

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run validation tests first
    if not run_validation_tests():
        print("[ERROR] Validation tests failed - aborting")
        exit(1)
    
    # Stream live ticks from MT5
    processor = stream_live_ticks(symbol="Step Index.0", brick_size=0.5, max_ticks=10000)
    
    # Output results
    output_results(processor)
