import pandas as pd


def close_position(position: pd.Series):
    position = position.copy()
    position[position.index == position.last_valid_index()] = 0.0
    return position
