# MatchboxNet
A reimplementation of the [MatchboxNet](https://www.isca-archive.org/interspeech_2020/majumdar20_interspeech.pdf) architecture for keyword spotting.

The model can be run in the command line with real-time inference by executing `run.py` (provided you have a working microphone and the necessary dependencies from the `requirements.txt` installed).

Future goals:
- Deploy the model on an embedded device like a Raspberry Pi.
- Update `train.py` to incorporate command line arguments via argparse.