import numpy as np
import copy

def compare_dict_array(da1, da2):
    """
    If the two dictionary-arrays is completely the same, return True;
    else, return False
    """
    if isinstance(da1, dict) and isinstance(da2, dict):
        if da1.keys() == da2.keys():
            for key in da1:
                if not compare_dict_array(da1[key], da2[key]): return False
            return True
        else: return False
    elif isinstance(da1, np.ndarray) and isinstance(da2, np.ndarray):
        return np.all(da1 == da2)
    else:
        return da1 == da2

def add_dict_array(da1, da2):
    if isinstance(da1, dict) and isinstance(da2, dict):
        if len(da1)  == 0:
            return da2
        elif len(da2) == 0:
            return da1
        else:
            assert da1.keys() == da2.keys()
            sum = dict()
            for key in da1:
                sum[key] = add_dict_array(da1[key], da2[key])
            return sum
    elif isinstance(da1, list) and isinstance(da2, list):
        assert len(da1) == len(da2)
        return [da1[i] + da2[i] for i in len(da1)]
    else: 
        return da1 + da2