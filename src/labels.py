import pandas as pd
import numpy as np

def compute_labels(df, horizon=10):
    # Assume df already has 'adj_high', 'adj_low', 'upper_barrier', and 'lower_barrier'
    # barriers are pre-calculated absolute price levels based on day t's close
    
    # 1. Build the forward-looking matrices using vector shifts
    # We grab the next 10 days of highs and lows all at once
    forward_highs = pd.concat([df['adj_high'].shift(-i) for i in range(1, horizon + 1)], axis=1)
    forward_lows = pd.concat([df['adj_low'].shift(-i) for i in range(1, horizon + 1)], axis=1)
    
    # 2. Create boolean masks: Did the forward high/low cross the barrier set at day t?
    # We use df[['upper_barrier']].values to broadcast day t's barrier across all 10 forward days
    hit_upper = forward_highs.values >= df[['upper_barrier']].values
    hit_lower = forward_lows.values <= df[['lower_barrier']].values
    
    # 3. Find the FIRST day a barrier is hit using argmax (which returns 0 if no True is found)
    # argmax stops at the first True. 
    upper_touch_idx = np.argmax(hit_upper, axis=1)
    lower_touch_idx = np.argmax(hit_lower, axis=1)
    
    # We need to distinguish between "touched on day 0 of the window" and "never touched"
    # so we explicitly check if ANY day was touched
    any_upper = np.any(hit_upper, axis=1)
    any_lower = np.any(hit_lower, axis=1)
    
    # 4. Resolve the race condition: which barrier was hit first?
    # Default to 0 (lower touch or timeout)[cite: 2]
    df['label'] = 0 
    df['t1'] = df.index + pd.Timedelta(days=horizon) # Default timeout date[cite: 2]
    
    for i in range(len(df)):
        u_idx = upper_touch_idx[i] if any_upper[i] else np.inf
        l_idx = lower_touch_idx[i] if any_lower[i] else np.inf
        
        if u_idx == np.inf and l_idx == np.inf:
            continue # Timeout, stays 0
            
        if u_idx < l_idx:
            df.iloc[i, df.columns.get_loc('label')] = 1
            # u_idx is 0-indexed from t+1, so add (u_idx + 1) days
            df.iloc[i, df.columns.get_loc('t1')] = df.index[i + u_idx + 1] 
        else:
            df.iloc[i, df.columns.get_loc('t1')] = df.index[i + l_idx + 1]

    return df