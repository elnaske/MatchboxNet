# MatchboxNet
A reimplementation of the [MatchboxNet](https://www.isca-archive.org/interspeech_2020/majumdar20_interspeech.pdf) architecture for keyword spotting.

The model can be run in the command line with real-time inference by executing `run.py` (provided you have a working microphone and the necessary dependencies from the `requirements.txt` installed).

# Requirements

## Inference

This project uses `sounddevice` for audio input, which requires PortAudio to be installed.

In Unix, run the following:
```bash
sudo apt update
sudo apt install libportaudio2 libportaudiocpp0 portaudio19-dev
```

Then install the required dependencies via pip. For embedded systems with less RAM (e.g. Raspberry Pi Zero 2 W) you might have to increase have to increase swap space to install larger packages like PyTorch (1GB should be enough).
```bash
pip install torch torchaudio sounddevice numpy tqdm
```

Depending on your microphone you may have to adjust the `ENERGY_THRESHOLD` in `run.py`. If the model only outputs `SIL`, the threshold is likely too high. Vice versa, if you only get false positives, the threshold is likely too low.

## Training

WIP

# Future goals
- Update `train.py` to incorporate command line arguments via argparse.