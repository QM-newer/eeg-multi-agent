# data/reader.py
"""
EEG数据读取器

- XLTEK/Natus（.eeg + .erd）：真实解析，见 data/xltek_reader.py
- EDF：通过 MNE 读取（需安装 mne）
"""

import numpy as np

from data.xltek_reader import XltekRecord


class EEGReader:
    """
    EEG数据读取基类
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.sfreq = None
        self.n_channels = None
        self.ch_names = None
        self.data = None

    def read_raw(self):
        """读取原始数据，返回 (data, sfreq, ch_names)"""
        raise NotImplementedError

    def get_duration(self):
        """获取记录时长（秒）"""
        if self.data is not None and self.sfreq is not None:
            return self.data.shape[1] / self.sfreq
        return None


class XLTEKReader(EEGReader):
    """
    XLTEK/Natus 格式 EEG 读取器（真实解析）

    一条记录 = 1 个 .eeg（信息头）+ 若干 .erd（delta 编码的原始信号分段）。
    记录中位数时长约 4 小时，全量解码代价高，因此按窗口读取：
        read_raw(skip_sec=60, duration_sec=600)
    """

    def __init__(self, eeg_path, erd_path=None, ent_path=None,
                 skip_sec=60.0, duration_sec=None):
        super().__init__(eeg_path)
        self.erd_path = erd_path
        self.ent_path = ent_path
        self.skip_sec = skip_sec
        self.duration_sec = duration_sec
        self.record = XltekRecord(eeg_path=eeg_path)
        self.sfreq = self.record.sfreq
        self.n_channels = self.record.n_channels
        self.ch_names = self.record.channel_names

    def read_raw(self, skip_sec=None, duration_sec=None, channels=None):
        """
        读取原始EEG数据

        参数:
            skip_sec: 跳过开头的秒数（None 用构造参数）
            duration_sec: 读取时长（秒，None = 读到记录结束）
            channels: 需要返回的通道索引（None = 全部）

        返回:
            (data, sfreq, ch_names)，data 形状 (n_channels, n_times)，单位 µV
        """
        skip = self.skip_sec if skip_sec is None else skip_sec
        dur = self.duration_sec if duration_sec is None else duration_sec
        self.data, self.sfreq, self.ch_names = self.record.read_window(
            skip_sec=skip, duration_sec=dur, channels=channels)
        self.n_channels = self.data.shape[0]
        return self.data, self.sfreq, self.ch_names


class EDFReader(EEGReader):
    """
    EDF格式读取器（使用MNE）
    """

    def read_raw(self):
        import mne
        raw = mne.io.read_raw_edf(self.file_path, preload=True)
        self.data = raw.get_data()
        self.sfreq = raw.info["sfreq"]
        self.ch_names = raw.ch_names
        self.n_channels = len(self.ch_names)
        return self.data, self.sfreq, self.ch_names


def get_reader(file_path, format_type="auto"):
    """
    根据文件格式自动选择读取器

    参数:
        file_path: 文件路径
        format_type: 格式类型，auto自动识别

    返回:
        EEGReader子类实例
    """
    if format_type == "auto":
        if file_path.endswith(".edf"):
            return EDFReader(file_path)
        elif file_path.endswith(".eeg"):
            return XLTEKReader(file_path)
        else:
            raise ValueError(f"无法识别的文件格式: {file_path}")
    elif format_type == "edf":
        return EDFReader(file_path)
    elif format_type == "xltek":
        return XLTEKReader(file_path)
    else:
        raise ValueError(f"不支持的格式: {format_type}")
