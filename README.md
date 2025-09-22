# MatchboxNet
A reimplementation of the [MatchboxNet](https://www.isca-archive.org/interspeech_2020/majumdar20_interspeech.pdf) architecture for keyword spotting.

The model can be run in the command line with real-time inference by executing `run.py` (provided you have a working microphone and the necessary dependencies from the `requirements.txt` installed).

# Installation

This project uses `sounddevice` for audio input, which requires PortAudio to be installed.

In Unix, run the following:
```bash
sudo apt update
sudo apt install libportaudio2 libportaudiocpp0 portaudio19-dev
```

Then install the required dependencies via pip:
```bash
pip install torch torchaudio sounddevice numpy tqdm
```

# Future goals
- Deploy the model on an embedded device like a Raspberry Pi.
- Update `train.py` to incorporate command line arguments via argparse.