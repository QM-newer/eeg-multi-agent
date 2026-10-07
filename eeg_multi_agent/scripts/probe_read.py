# scripts/probe_read.py
"""
真实数据读取冒烟测试

做三件事：
  1) 解析一条真实记录的 .erd 头，确认采样率/通道数/headbox 型号
  2) 用自己的解码器读一小段信号，与 XltekDataReader 参考实现逐点交叉校验
  3) 输出各通道幅值统计（判读单位是否正确、识别 DC 通道）与解码速度（估算全库耗时）

用法：
    python scripts/probe_read.py                      # 取 record_index 里第一条记录
    python scripts/probe_read.py --seconds 30         # 读 30 秒
    python scripts/probe_read.py --no-validate        # 跳过与参考实现的交叉校验
    python scripts/probe_read.py --eeg "E:/.../xxx.eeg"
"""

import os
import sys
import csv
import mmap
import time
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_CONFIG                     # noqa: E402
from data.xltek_reader import (XltekRecord, read_erd_header, decode_erd,  # noqa: E402
                               DATA_OFFSET)

READER_DIR = r"E:\eeg_organize_tmp\XltekDataReader"   # 参考实现（仅用于交叉校验）


def pick_default_record():
    idx = DATA_CONFIG["record_index"]
    if not os.path.exists(idx):
        raise FileNotFoundError(f"请先构建记录主表: {idx}（python -m data.build_record_index）")
    with open(idx, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["erd_paths"] and os.path.exists(r["eeg_path"]):
                return r["eeg_path"], r["record_id"], r.get("label_name", "")
    raise RuntimeError("record_index 中没有可用记录")


def validate_with_reference(eeg_path, n_packets, header):
    """
    与 XltekDataReader 参考实现做"结构级"交叉校验：
    比较每个数据包解出的通道集合，以及该包结束后游标落在的字节偏移。

    说明：参考实现没有做「增量是有符号数」的还原（会把信号解成随机游走），
    所以数值上无法逐点比对；但只要逐包的通道集合与字节偏移完全一致，
    就说明掩码/降采样分组/绝对值标记的解析是正确的。
    """
    if not os.path.isdir(READER_DIR):
        return "跳过（未找到参考实现目录 %s）" % READER_DIR

    cwd = os.getcwd()
    sys.path.insert(0, READER_DIR)
    try:
        os.chdir(READER_DIR)
        from file_loader.file_types.erd_file import ERDLoader
        from file_loader.utils.raw_data_packet import RawDataObject
        from file_loader.utils.byte_buffer import ByteBuffer

        rec = XltekRecord(eeg_path=eeg_path)
        seg = rec.segments()[0]

        class _Stop(Exception):
            pass

        class LimitedBuffer(ByteBuffer):
            """只加载前 limit 字节，接近末尾时干净地停止（避免参考实现在截断处报错）"""

            def __init__(self, path, limit):
                with open(path, "rb") as f:
                    self.s = f.read(limit)
                self.cursor = DATA_OFFSET      # 头已解析，数据包从 DATA_OFFSET 开始

            def read(self, read_format=None, num_read=1, null_terminate=True):
                if self.cursor >= len(self.s) - 64:
                    raise _Stop()
                return ByteBuffer.read(self, read_format, num_read, null_terminate)

        # 1) 先用参考实现只解析文件头（屏蔽掉整文件解码）
        orig_load = RawDataObject.load_file
        RawDataObject.load_file = lambda self, buf: None
        loader = ERDLoader(seg)
        loader.load()
        RawDataObject.load_file = orig_load

        # 2) 再用限长缓冲解前若干包
        limit = DATA_OFFSET + int(n_packets * header.num_channels * 4)
        obj = RawDataObject(loader)
        # 参考实现用 None 初始化基准值，遇到"绝对值之前的第一个增量"就会崩，这里补 0
        obj.last_channel_value = [0.0] * loader.num_channels
        try:
            obj.load_file(LimitedBuffer(seg, limit))
        except _Stop:
            pass

        # 3) 自己的解码器跑同样多的包
        with open(seg, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            _, _, packets = decode_erd(mm, header, n_samples=n_packets,
                                       collect_packets=True)

        n = min(len(packets), len(obj.channels_list), n_packets)
        if n == 0:
            return "参考实现未解出数据包"

        # 参考实现把绝对值通道追加在增量通道之后，顺序不同，故只比较集合与偏移
        ch_mis, off_mis = 0, 0
        for t in range(n):
            if sorted(obj.channels_list[t]) != sorted(packets[t][0]):
                ch_mis += 1
            if obj.packet_file_offset[t] != packets[t][1]:
                off_mis += 1
        ok = (ch_mis == 0 and off_mis == 0)
        return (f"比对前 {n} 个数据包：通道集合不一致 {ch_mis} 个，"
                f"字节偏移不一致 {off_mis} 个  "
                f"{'✅ 解析结构一致' if ok else '❌ 解析结构不一致'}")
    except Exception as e:                 # noqa: BLE001
        import traceback
        return f"参考实现校验失败: {type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}"
    finally:
        os.chdir(cwd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eeg", default=None, help="记录的 .eeg 路径")
    ap.add_argument("--seconds", type=float, default=10.0, help="读取时长（秒）")
    ap.add_argument("--skip", type=float, default=0.0, help="跳过开头秒数")
    ap.add_argument("--validate-samples", type=int, default=2000, help="交叉校验的采样点数")
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")

    if args.eeg:
        eeg_path, record_id, label = args.eeg, os.path.basename(args.eeg), ""
    else:
        eeg_path, record_id, label = pick_default_record()

    print("=" * 70)
    print("记录:", record_id, " 标签:", label or "-")
    print("路径:", eeg_path)
    print("=" * 70)

    rec = XltekRecord(eeg_path=eeg_path)
    header = rec.header()
    print("\n[1] 文件头解析")
    print("   ", header.summary())
    print("    分段数:", len(rec.segments()))
    for s in rec.segments():
        print("      ", os.path.basename(s), "%.1f MB" % (os.path.getsize(s) / 1e6))
    print("    通道名:", rec.channel_names)

    print(f"\n[2] 读取窗口 skip={args.skip}s duration={args.seconds}s")
    t0 = time.time()
    data, sfreq, ch_names = rec.read_window(skip_sec=args.skip, duration_sec=args.seconds)
    dt = time.time() - t0
    print(f"    数据形状: {data.shape}  采样率: {sfreq} Hz  实际时长: "
          f"{data.shape[1]/sfreq:.1f}s  耗时: {dt:.1f}s")
    if data.shape[1] == 0:
        print("    [失败] 未解出任何采样点")
        return

    speed = data.shape[1] / dt if dt > 0 else 0
    print(f"    解码速度: {speed:,.0f} 采样点/秒（{speed/sfreq:.1f}× 实时）")

    print("\n[3] 各通道幅值统计（单位 µV）")
    print("    ch    名称      std      ptp      均值")
    for i, name in enumerate(ch_names):
        x = data[i]
        print(f"    {i:2d}   {name:<6s} {np.std(x):8.2f} {np.ptp(x):8.2f} {np.mean(x):9.2f}")
    med = np.median([np.std(data[i]) for i in range(data.shape[0])])
    print(f"    中位 std = {med:.2f} µV")
    if med > 500:
        print("    [提示] 幅值明显偏大，缩放系数可能多乘了 2^discardbits "
              "（可用 apply_discard_in_factor=False 复测）")
    elif med < 1:
        print("    [提示] 幅值明显偏小，请检查缩放系数")
    else:
        print("    [提示] 幅值量级符合常规 EEG（约 5~100 µV）")

    # 平稳性检查：若增量符号/缩放搞错，信号会随机游走（前后半段均值与 std 差异巨大）
    half = data.shape[1] // 2
    if half > 100:
        s1, s2 = np.std(data[:, :half]), np.std(data[:, half:2 * half])
        d1, d2 = np.mean(data[:, :half]), np.mean(data[:, half:2 * half])
        print(f"    平稳性: 前半 std={s1:.2f} 后半 std={s2:.2f} | "
              f"前半均值={d1:.2f} 后半均值={d2:.2f}")

    # 频谱合理性（EEG 能量应集中在低频，且不应是白噪声）
    x = data[0].astype(np.float64)
    if len(x) >= 512:
        spec = np.abs(np.fft.rfft(x - x.mean())) ** 2
        freqs = np.fft.rfftfreq(len(x), 1.0 / sfreq)
        bands = {"delta(0.5-4)": (0.5, 4), "theta(4-8)": (4, 8), "alpha(8-13)": (8, 13),
                 "beta(13-30)": (13, 30), "gamma(30-45)": (30, 45)}
        tot = spec[(freqs >= 0.5) & (freqs <= 45)].sum() or 1.0
        print("    频带能量占比（Ch1）: " + "  ".join(
            f"{k}={spec[(freqs >= lo) & (freqs < hi)].sum()/tot*100:4.1f}%"
            for k, (lo, hi) in bands.items()))

    if not args.no_validate:
        print("\n[4] 与 XltekDataReader 参考实现交叉校验（解析结构）")
        print("   ", validate_with_reference(eeg_path, args.validate_samples, rec.header()))

    print("\n[5] 全库耗时估算（按 %.0f 秒/条 × 1598 条）" % args.seconds)
    print(f"    单进程: {dt*1598/3600:.1f} 小时   |   8 进程并行: {dt*1598/3600/8:.1f} 小时")


if __name__ == "__main__":
    main()
