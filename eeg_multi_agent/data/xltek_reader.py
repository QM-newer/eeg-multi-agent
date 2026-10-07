# data/xltek_reader.py
"""
XLTEK / Natus NeuroWorks 真实数据解析器（.eeg 头 + .erd 原始信号）

背景
----
一条记录由多个文件组成：
  .eeg  患者/检查信息（StudyInfo，含 headbox 型号、通道数、采样率）
  .erd  原始信号（delta 编码的数据包流；长记录会被切成 _001 / _002 ... 多个分段）
  .etc  erd 的目录（table of content）
  .ent  标注/事件
本项目设备：headbox_type=26，采样率 500Hz，32 AC + 3 DC = 35 通道，deltabits=8，discardbits=6

数据格式（schema 9，与 XltekDataReader 的 data_template 一致）
-------------------------------------------------------------
  通用头 352 字节  +  erd 专有头 8304 字节  =>  数据包从 8656 字节开始
每个数据包 = 1 个事件字节 [+1 个频率字节] + ceil(n_ch/8) 个掩码字节 + 各通道 1~2 字节增量
  值 0xFFFF 是"绝对值"标记，后跟 4 字节有符号整数（小端）
解析必须从头顺序进行（增量编码），无法随机 seek，因此按"窗口"读取时仍需顺序解码前面部分。

用法
----
    from data.xltek_reader import XltekRecord
    rec = XltekRecord(r"E:/脑电数据按姓名整理/于若水/于~ 若水_bb1f7703-....eeg")
    data, sfreq, ch_names = rec.read_window(skip_sec=60, duration_sec=600)
    # data: (n_channels, n_times)，单位 µV

与参考实现 XltekDataReader 的三处关键差异（均已在本数据集上实测验证）
------------------------------------------------------------------
1. 2 字节增量是**小端**（参考实现按 big 读，会得到发散的随机游走）
2. 参考实现既对读数左移 discardbits、又在系数里乘了 2^discardbits（重复 64 倍），
   这里只保留一次：物理量 = (原始值 << discardbits) * 8711/(2^21-0.5)
3. 通道首个增量之前可能没有绝对值基准（参考实现会抛 None + float 异常），
   此处以首个增量作为基准

验证依据：
- 与参考实现逐包比对 3000 包：通道集合与字节偏移 0 处不一致（解析结构一致）
- 绝对值锚点误差（逐通道中位）0.276µV，重建信号 lag-1 自相关 0.991（平滑如真实脑电）
- 幅值：带通后 RMS 数十 µV、Ch33~Ch35 恒为 DC 值、频谱 delta 主导（儿童脑电典型形态）
- 自洽性：16 位增量上限 32767 << 6 ≈ 2^21，正好对应满量程 8711 µV
"""

import os
import mmap
import struct
import numpy as np

# ------------------------------------------------------------------
# 常量（来自 data_templates/erd/file_schema_9/data_template.json）
# ------------------------------------------------------------------
GENERIC_HEADER_SIZE = 352          # read_checkpoint_1
ERD_HEADER_SIZE = 8304             # 专有头大小
DATA_OFFSET = GENERIC_HEADER_SIZE + ERD_HEADER_SIZE   # = 8656，read_checkpoint_2

ERD_FILE_TYPE_GUID = (-905246832, 298899349, -1610599761, -1521198300)
SHRT_MAX = 32767
FFFF = 65535

# 频率字节位图：最低置位位 -> 该通道组的降采样因子（-1 表示全部通道）
BITMAP_TRANSLATE = {0: 2, 1: 4, 2: 5, 3: 10, 4: 20, 5: 50, 7: -1}

# headbox_type=26 的转换系数（µV/LSB）：8711 / (2^21 - 0.5)
# 样本是 22 位有符号（±2^21 对应 ±8711µV 满量程）；增量以 deltabits=8/16 位有符号数存储，
# 且丢弃了 discardbits=6 个低位，故：物理量 µV = (原始值 << discardbits) * LSB
# 自洽性验证：16 位增量最大值 32767 << 6 = 2097088 ≈ 2^21，正好对应满量程 8711 µV。
_UV_LSB_BASE = 8711.0 / (2.0 ** 21 - 0.5)


class ERDHeader(object):
    """erd 文件头（解析后即可知道通道数、采样率、缩放系数）"""

    __slots__ = ("sample_freq", "num_channels", "deltabits", "discardbits",
                 "phys_chan", "headbox_type", "headbox_sn", "shorted",
                 "frequency_factor", "channel_names", "channel_factors",
                 "apply_discard_in_factor", "abs_escape")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    @property
    def n_mask_bytes(self):
        return (self.num_channels + 7) // 8

    def summary(self):
        return (f"sfreq={self.sample_freq:g}Hz  n_channels={self.num_channels}  "
                f"deltabits={self.deltabits}  discardbits={self.discardbits}  "
                f"headbox_type={self.headbox_type}")


def read_erd_header(path, apply_discard_in_factor=False):
    """解析 .erd 文件头（只读前 8656 字节，开销极小）"""
    with open(path, "rb") as f:
        head = f.read(DATA_OFFSET)
    if len(head) < DATA_OFFSET:
        raise ValueError(f"文件过短，不是有效的 .erd: {path}")

    # 通用头校验
    # 注意：上游模板给出的第一个 GUID 常数是 -905246832，而真实 .erd 文件为 -905246831
    # （差 1，见 E 盘实测；.eeg 文件则是 -905246832）。因此只校验后 3 个 int 与 file_schema。
    guid = struct.unpack_from("<4i", head, 0)
    file_schema, base_schema = struct.unpack_from("<hh", head, 16)
    if tuple(guid[1:]) != ERD_FILE_TYPE_GUID[1:]:
        raise ValueError(
            f"文件类型 GUID 不匹配（不是 NeuroWorks .erd）: {path}  guid={guid}")
    if file_schema != 9:
        raise ValueError(f"仅支持 file_schema=9，实际为 {file_schema}: {path}")
    if base_schema != 1:
        raise ValueError(f"仅支持 base_schema=1，实际为 {base_schema}: {path}")

    off = GENERIC_HEADER_SIZE
    sample_freq = struct.unpack_from("<d", head, off)[0]; off += 8
    num_channels = struct.unpack_from("<i", head, off)[0]; off += 4
    deltabits = struct.unpack_from("<i", head, off)[0]; off += 4
    phys_chan = list(struct.unpack_from("<1024i", head, off)); off += 4096
    headbox_type = struct.unpack_from("<4i", head, off)[0]; off += 16
    headbox_sn = struct.unpack_from("<4i", head, off)[0]; off += 16
    off += 40 + 10 + 10                      # 三个版本字符串
    discardbits = struct.unpack_from("<i", head, off)[0]; off += 4
    shorted = list(struct.unpack_from("<1024h", head, off)); off += 2048
    frequency_factor = list(struct.unpack_from("<1024h", head, off)); off += 2048

    if deltabits != 8:
        raise ValueError(f"仅支持 deltabits=8，实际为 {deltabits}: {path}")

    # 通道名：headbox 26 的设备未在文件中记录解剖命名，只有 Ch1..ChN
    channel_names = ["Ch%d" % (i + 1) for i in range(num_channels)]

    # 转换系数（µV/LSB）
    if headbox_type == 26:
        factor = _UV_LSB_BASE * (2.0 ** discardbits if apply_discard_in_factor else 1.0)
        channel_factors = [factor] * num_channels
    else:
        raise ValueError(f"未支持的 headbox_type={headbox_type}（当前仅验证过 26）: {path}")

    return ERDHeader(
        sample_freq=float(sample_freq), num_channels=int(num_channels),
        deltabits=int(deltabits), discardbits=int(discardbits),
        phys_chan=phys_chan[:num_channels], headbox_type=int(headbox_type),
        headbox_sn=int(headbox_sn),
        shorted=[1 if s else 0 for s in shorted[:num_channels]],
        frequency_factor=frequency_factor[:num_channels],
        channel_names=channel_names, channel_factors=channel_factors,
        apply_discard_in_factor=apply_discard_in_factor,
        # 0xFFFF = 「绝对值转义」标记：该通道本包无增量，改用包尾的 4 字节绝对值
        abs_escape=True,
    )


def _build_channel_groups(header):
    """按降采样因子分组：{factor: [channel_id, ...]}，返回 (groups, has_freq)"""
    groups = {}
    has_freq = False
    for ch, ff in enumerate(header.frequency_factor):
        if ff != SHRT_MAX:
            has_freq = True
            groups.setdefault(ff, []).append(ch)
    return groups, has_freq


def decode_erd(buf, header, start_sample=0, n_samples=None, channels=None,
               progress=None, collect_packets=False):
    """
    顺序解码 .erd 数据包（delta 编码，必须从文件头开始）

    参数:
        buf: bytes / mmap，整个 .erd 文件的内容
        header: ERDHeader
        start_sample: 跳过前 start_sample 个采样点（仍需解码，只是不保存）
        n_samples: 需要保存的采样点数；None 表示解码到文件结束
        channels: 需要保留的通道索引（None = 全部；其余通道仍参与解码以推进游标）
        progress: 可选回调 progress(n_decoded_samples)
        collect_packets: True 时额外返回每个包的 (通道号元组, 包结束后的字节偏移)，
                         用于与参考实现做"结构级"交叉校验

    返回:
        (data, n_decoded)              或  (data, n_decoded, packets)
            data: (n_keep_channels, n_saved) float64，单位 µV
            n_decoded: 实际解码出的采样点总数
    """
    n_ch = header.num_channels
    if channels is None:
        keep_idx = list(range(n_ch))
    else:
        keep_idx = list(channels)
    n_keep = len(keep_idx)

    groups, has_freq = _build_channel_groups(header)
    shorted = set(ch for ch, s in enumerate(header.shorted) if s)
    discard = header.discardbits
    abs_escape = getattr(header, "abs_escape", False)
    factors = np.asarray(header.channel_factors, dtype=np.float64)
    n_mask = header.n_mask_bytes

    # 每个 subsample 取值下参与本包的通道（顺序固定）
    incl = {}
    for sub in set(list(groups.keys()) + [-1]):
        if sub == -1:
            incl[sub] = [ch for ch in range(n_ch) if ch not in shorted]
        else:
            incl[sub] = [ch for ch in groups.get(sub, []) if ch not in shorted]

    # 预先算好每个 subsample 下参与本包的通道（通道号可能不连续，掩码必须按通道号取位）
    incl_arr = {s: np.array(v, dtype=np.int64) for s, v in incl.items() if v}

    if n_samples is None:
        out = np.empty((n_keep, max(1, len(buf) // n_ch)), dtype=np.float64)
    else:
        out = np.empty((n_keep, int(n_samples)), dtype=np.float64)

    last = np.zeros(n_ch, dtype=np.float64)
    seen = np.zeros(n_ch, dtype=bool)
    saved = 0
    decoded = 0
    pos = DATA_OFFSET
    size = len(buf)
    packets = [] if collect_packets else None

    while pos < size:
        # --- 事件字节（本阶段不使用事件信息）---
        pos += 1

        # --- 可选的频率字节 ---
        if has_freq:
            if pos >= size:
                break
            b = buf[pos]; pos += 1
            i = 0
            while i < 9:
                if (b & 1) and (i in BITMAP_TRANSLATE):
                    break
                b >>= 1
                i += 1
            if i >= 9:
                break                      # 文件截断/异常，结束解码
            subsample = BITMAP_TRANSLATE[i]
        else:
            subsample = -1

        chs = incl_arr.get(subsample)
        if chs is None or len(chs) == 0:
            # 该包无有效通道，仍需推进掩码字节
            pos += n_mask
            decoded += 1
            if collect_packets:
                packets.append(((), pos))
            continue

        # --- 掩码字节：判断哪些通道用 2 字节增量（按通道号取位，通道号可能不连续）---
        if pos + n_mask > size:
            break
        bits = np.frombuffer(buf, dtype=np.uint8, count=n_mask, offset=pos)
        pos += n_mask
        dbl_full = np.unpackbits(bits, bitorder='little')[:n_ch]
        double = dbl_full[chs].astype(bool)

        # --- 各通道增量 ---
        # 重要：0xFFFF 是"绝对值转义"标记，但其 4 字节绝对值统一排在所有通道的
        # 增量字节之后（见参考实现 RawDataObject.load_file），不能就地读取，否则字节错位。
        n_c = len(chs)
        total = int(np.where(double, 2, 1).sum())
        if pos + total > size:
            break

        if not double.any():
            # 快路径：全部 1 字节，不可能出现 0xFFFF 标记，向量化处理
            raw = np.frombuffer(buf, dtype=np.uint8, count=n_c, offset=pos)
            pos += n_c
            # 增量是 deltabits 位有符号数（8 位 -> -128..127），必须转回有符号
            signed = np.where(raw > 127, raw.astype(np.int64) - 256, raw.astype(np.int64))
            deltas = (signed << discard) * factors[chs]
            last[chs] = np.where(seen[chs], last[chs] + deltas, deltas)
            seen[chs] = True
        else:
            p = pos
            abs_chs = []
            delta_chs = []
            delta_vals = []
            for j in range(n_c):
                ch = chs[j]
                if double[j]:
                    # 实测：2 字节增量是小端（与参考实现写的 big 相反）
                    v = int.from_bytes(buf[p:p + 2], 'little'); p += 2
                    if abs_escape and v == FFFF:
                        abs_chs.append(ch)
                        continue
                    if v > 32767:
                        v -= 65536           # 16 位有符号
                else:
                    v = buf[p]; p += 1
                    if v > 127:
                        v -= 256             # 8 位有符号
                delta_chs.append(ch)
                delta_vals.append(v)
            pos = p

            # 绝对值：4 字节有符号整数（小端），即完整采样值
            if abs_chs:
                if pos + 4 * len(abs_chs) > size:
                    break
                for ch in abs_chs:
                    v = struct.unpack_from("<i", buf, pos)[0]; pos += 4
                    last[ch] = (v << discard) * factors[ch]
                    seen[ch] = True

            # 增量：累加到上一采样值
            for ch, v in zip(delta_chs, delta_vals):
                delta = (v << discard) * factors[ch]
                if seen[ch]:
                    last[ch] += delta
                else:
                    # 首个增量之前没有绝对值基准，则以该值作为基准
                    last[ch] = delta
                    seen[ch] = True

        decoded += 1
        if collect_packets:
            packets.append((tuple(int(c) for c in chs), pos))

        if decoded > start_sample and (n_samples is None or saved < n_samples):
            if saved >= out.shape[1]:
                out = np.concatenate(
                    [out, np.empty((n_keep, max(n_samples or 1, out.shape[1])), dtype=np.float64)],
                    axis=1)
            out[:, saved] = last[keep_idx]
            saved += 1

        if n_samples is not None and saved >= n_samples:
            break

        if progress is not None and decoded % 200000 == 0:
            progress(decoded)

    if collect_packets:
        return out[:, :saved], decoded, packets
    return out[:, :saved], decoded


def decode_erd_file(path, start_sample=0, n_samples=None, channels=None,
                    apply_discard_in_factor=False):
    """打开 .erd 文件（mmap，避免整文件进内存）并解码"""
    header = read_erd_header(path, apply_discard_in_factor=apply_discard_in_factor)
    with open(path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            data, n = decode_erd(mm, header, start_sample=start_sample,
                                 n_samples=n_samples, channels=channels)
    return data, header, n


class XltekRecord(object):
    """一条完整记录（一个 .eeg + 若干 .erd 分段）"""

    def __init__(self, eeg_path=None, folder=None, record_id=None,
                 apply_discard_in_factor=False):
        if eeg_path is None:
            if folder is None or record_id is None:
                raise ValueError("需要 eeg_path 或 (folder, record_id)")
            eeg_path = os.path.join(folder, record_id + ".eeg")
        self.eeg_path = os.path.normpath(eeg_path)
        self.folder = os.path.dirname(self.eeg_path)
        self.record_id = os.path.splitext(os.path.basename(self.eeg_path))[0]
        self.apply_discard_in_factor = apply_discard_in_factor
        self._header = None
        self._segments = None

    # ---------- 分段文件 ----------
    def segments(self):
        """该记录的所有 .erd 分段，按时间顺序排列"""
        if self._segments is None:
            segs = []
            if os.path.isdir(self.folder):
                for name in os.listdir(self.folder):
                    if not name.lower().endswith(".erd"):
                        continue
                    stem = name[:-4]
                    if stem == self.record_id:
                        segs.append((0, name))
                    elif stem.startswith(self.record_id + "_"):
                        suf = stem[len(self.record_id) + 1:]
                        try:
                            segs.append((int(suf), name))
                        except ValueError:
                            segs.append((999, name))
            segs.sort(key=lambda x: x[0])
            self._segments = [os.path.join(self.folder, n) for _, n in segs]
        return self._segments

    # ---------- 头信息 ----------
    def header(self):
        if self._header is None:
            segs = self.segments()
            if not segs:
                raise FileNotFoundError(f"未找到 .erd 分段: {self.folder}")
            self._header = read_erd_header(
                segs[0], apply_discard_in_factor=self.apply_discard_in_factor)
        return self._header

    @property
    def sfreq(self):
        return self.header().sample_freq

    @property
    def n_channels(self):
        return self.header().num_channels

    @property
    def channel_names(self):
        return list(self.header().channel_names)

    # ---------- 读取窗口 ----------
    def read_window(self, skip_sec=60.0, duration_sec=600.0, channels=None):
        """
        读取一段连续信号（跨 .erd 分段自动拼接）

        参数:
            skip_sec: 从记录开头跳过的秒数（解码但不保存，用于避开准备期伪迹）
            duration_sec: 读取时长（秒）；None = 读到记录结束
            channels: 需要返回的通道索引（None = 全部）

        返回:
            (data, sfreq, ch_names)
                data: (n_channels, n_times) float32，单位 µV
        """
        header = self.header()
        sf = header.sample_freq
        start_sample = int(round(skip_sec * sf))
        n_samples = None if duration_sec is None else int(round(duration_sec * sf))

        parts = []
        got = 0
        for seg in self.segments():
            need = None if n_samples is None else max(0, n_samples - got)
            if need == 0:
                break
            data, _ = decode_erd_file(
                seg, start_sample=start_sample, n_samples=(need if need else None),
                channels=channels,
                apply_discard_in_factor=self.apply_discard_in_factor,
            )[:2]
            # 每个分段独立解码：后续分段不再跳过开头
            start_sample = 0
            if data.shape[1] > 0:
                parts.append(data)
                got += data.shape[1]
        if not parts:
            return (np.zeros((len(channels) if channels else header.num_channels, 0),
                             dtype=np.float32), sf, self.channel_names)

        out = np.concatenate(parts, axis=1)
        if n_samples is not None:
            out = out[:, :n_samples]
        names = self.channel_names
        if channels is not None:
            names = [names[c] for c in channels]
        return out.astype(np.float32), sf, names
