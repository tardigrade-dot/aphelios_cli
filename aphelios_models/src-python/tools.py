#!/Volumes/sw/conda_envs/vibevoice/bin/python
from pydub import AudioSegment

def cut_wav(input_path: str, output_path: str, t1: float, t2: float):
    """
    截取 wav 文件中 t1 到 t2 秒之间的音频并保存
    :param input_path: 输入 wav 文件路径
    :param output_path: 输出 wav 文件路径
    :param t1: 起始时间（秒）
    :param t2: 结束时间（秒）
    """
    if t1 < 0 or t2 <= t1:
        raise ValueError("时间参数不合法")

    audio = AudioSegment.from_wav(input_path)

    # pydub 使用毫秒
    start_ms = int(t1 * 1000)
    end_ms = int(t2 * 1000)

    segment = audio[start_ms:end_ms]
    segment.export(output_path, format="wav")


# 示例
cut_wav("/Users/larry/Documents/resources/qinsheng-5s.wav", "/Users/larry/Documents/resources/qinsheng-4s.wav", 0.0, 3.8)