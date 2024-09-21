import os
import copy
from typing import Dict

from paddle_ocr.postprocess.db_postprocess import DBPostProcess
from paddle_ocr.operators import *


def transform(data, ops=None):
    """ transform """
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data


def create_operators(op_param_list, global_config=None):
    """
    create operators based on the config
    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(op_param_list, list), ('operator config should be a list')
    ops = []
    for operator in op_param_list:
        assert isinstance(operator,
                          dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        if global_config is not None:
            param.update(global_config)
        op = eval(op_name)(**param)
        ops.append(op)
    return ops


def build_post_process(config: Dict, global_config=None):

    config = copy.deepcopy(config)
    post_process_name = config.pop('name')
    assert post_process_name == "DBPostProcess"
    if global_config is not None:
        config.update(global_config)
    return DBPostProcess(**config)
