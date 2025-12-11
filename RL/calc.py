from __future__ import annotations

import numpy as np
import scipy.stats as stats

def confidence_interval(d:list[float], α:int = 0.05) -> tuple[float, float]:
    
    # convert d to numpy array
    d = np.array(d)

    # get length of data
    n = len(d)

    # Standard Error
    SE = np.std( d, ddof=1 ) / np.sqrt(n)
    
    # t_critical
    t = stats.t.ppf(1 - α/2, n - 1)
    
    mean = np.mean( d ) 
    halfwidth = t * SE
    
    return mean, halfwidth