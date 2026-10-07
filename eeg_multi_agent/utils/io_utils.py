# utils/io_utils.py
"""
文件读写工具
"""

import os
import json
import pickle

import numpy as np


def ensure_dir(path):
    """确保目录存在，不存在则创建（空路径直接跳过）"""
    if path:
        os.makedirs(path, exist_ok=True)


def save_json(data, filepath):
    """保存为JSON"""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(filepath):
    """加载JSON"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(data, filepath):
    """保存为pickle"""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_pickle(filepath):
    """加载pickle"""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_numpy(array, filepath):
    """保存numpy数组"""
    ensure_dir(os.path.dirname(filepath))
    np.save(filepath, array)


def load_numpy(filepath):
    """加载numpy数组"""
    return np.load(filepath)


def list_files_with_extension(root_dir, extension):
    """
    递归列出目录下所有指定后缀的文件

    参数:
        root_dir: 根目录
        extension: 文件后缀，如 '.eeg'

    返回:
        list: 文件路径列表
    """
    file_list = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                file_list.append(os.path.join(root, file))
    return file_list
