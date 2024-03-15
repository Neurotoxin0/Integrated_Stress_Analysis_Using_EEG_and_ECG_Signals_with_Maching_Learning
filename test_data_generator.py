import config
import numpy as np
import pandas as pd


np.random.seed(42)  # For reproducibility


def generate():
    """
    Simulate / Generate a single input example as if it has been processed and standardized
    contain 60 numerical features

    :return: A simulated input example
    """
    

    # Generate a random example with values that might represent standardized EEG and ECG features
    simulated_input_example = np.random.normal(0, 1, (1, len(config.features)))

    # Reshape for the format similar to the final input structure provided
    simulated_input_example = pd.DataFrame(simulated_input_example, columns = config.features)

    return simulated_input_example



if __name__ == "__main__":
    print(generate())
