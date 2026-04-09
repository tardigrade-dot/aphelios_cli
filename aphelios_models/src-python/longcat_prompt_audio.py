#!/usr/bin/env python3
import argparse
import sys

import librosa
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Load LongCat prompt audio with librosa")
    parser.add_argument("--audio_path", required=True)
    parser.add_argument("--sample_rate", type=int, required=True)
    args = parser.parse_args()

    audio, _ = librosa.load(args.audio_path, sr=args.sample_rate, mono=True)
    sys.stdout.buffer.write(np.asarray(audio, dtype="<f4").tobytes())


if __name__ == "__main__":
    main()
