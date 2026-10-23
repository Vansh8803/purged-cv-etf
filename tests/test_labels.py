import pandas as pd
import numpy as np
from src.labels import compute_labels

def test_triple_barrier_scenarios():
    # 1. Create a 15-day trading calendar (skips weekends)
    dates = pd.bdate_range(start='2026-10-01', periods=15)
    
    # 2. Build the toy dataframe with a flat baseline of 100
    df = pd.DataFrame(index=dates)
    df['adj_high'] = 101.0
    df['adj_low'] = 99.0
    
    # 3. Set static barriers at 105 and 95 (representing +/- 5% volatility)
    df['upper_barrier'] = 105.0
    df['lower_barrier'] = 95.0
    
    # --- SCENARIO A: Upper barrier hit (Price shoots up on day 3) ---
    df.loc[dates[3], 'adj_high'] = 106.0
    
    # --- SCENARIO B: Lower barrier hit (Price tanks on day 7) ---
    df.loc[dates[7], 'adj_low'] = 94.0
    
    # --- SCENARIO C: Timeout (Price stays flat from day 10 onwards) ---
    # (No manual intervention needed, highs/lows stay at 101/99)
    
    # 4. Run the vectorised labelling function
    df_labelled = compute_labels(df, horizon=10)
    
    # 5. Assert Scenario A (Evaluated from day 0)
    # Day 0 looks forward. Day 3 is within the 10-day window and hits the upper barrier.
    assert df_labelled.loc[dates[0], 'label'] == 1, "Failed to label upper barrier touch"
    assert df_labelled.loc[dates[0], 't1'] == dates[3], "Failed to record correct upper touch date"
    
    # 6. Assert Scenario B (Evaluated from day 5)
    # Day 5 looks forward. Day 7 hits the lower barrier first.
    assert df_labelled.loc[dates[5], 'label'] == 0, "Failed to label lower barrier touch as 0"
    assert df_labelled.loc[dates[5], 't1'] == dates[7], "Failed to record correct lower touch date"
    
    # 7. Assert Scenario C (Evaluated from day 10)
    # Day 10 looks forward. Window ends at day 14 (if horizon is limited to available data) 
    # or hits the timeout constraint. Neither barrier is touched.
    # Note: If your function pads missing future data, adjust the expected timeout date.
    assert df_labelled.loc[dates[10], 'label'] == 0, "Failed to label timeout as 0"