"""
@Description :   
@Author      :   Yang Xu
"""

import config
import numpy as np
import pandas as pd


def generate(rdm_seed = 42):
    """
    Simulate / Generate a single input example as if it has been processed and standardized

    :return: A simulated input example
    """
    min_val, max_val = 0, 100
    np.random.seed(rdm_seed)
    
    # Generate a random example with values that might represent standardized EEG and ECG features
    simulated_input_data = np.random.uniform(min_val, max_val, (1, len(config.features)))
    
    # Reshape for the format similar to the final input structure provided
    simulated_input_data = pd.DataFrame(simulated_input_data, columns = config.features)

    return simulated_input_data



if __name__ == "__main__":
    print(generate())
