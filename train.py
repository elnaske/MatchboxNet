import os
import json
from collections import defaultdict
from torch.utils.data import DataLoader

from dataset import KeywordsDataset, collate_data_aug
from trainer import Trainer
from model import MatchboxNet

batch_size = 256
n_epochs = 200
n_runs = 1

keywords = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

if not os.path.isdir('data'):
    os.mkdir('data')

# All classes
# train_v2 = KeywordsDataset('training')
# dev_v2 = KeywordsDataset('validation')
# test_v2 = KeywordsDataset('testing', keywords = keywords, silence = True)

# 12 Classes
train_v2 = KeywordsDataset('training', keywords = keywords, silence = True)
dev_v2 = KeywordsDataset('validation', keywords = keywords, silence = True)
test_v2 = KeywordsDataset('testing', keywords = keywords, silence = True)

label_mapper_v2 = train_v2.label_mapper
class_weights = train_v2.get_class_weights()

train_dataloader = DataLoader(train_v2, batch_size=batch_size, shuffle=True, collate_fn=collate_data_aug)
dev_dataloader = DataLoader(dev_v2, batch_size=len(dev_v2), shuffle=False)
test_dataloader = DataLoader(test_v2, batch_size=len(test_v2), shuffle=False)


results = defaultdict(dict)

for i in range(n_runs):
    model = MatchboxNet(in_channels=64,
                        n_classes=train_v2.n_classes,
                        B=3,
                        S=2)

    trainer = Trainer(model=model, class_weights=class_weights)

    trainer.fit(train_dataloader,
                dev_dataloader,
                epochs=n_epochs,
                max_lr=0.05,
                min_lr=0.001,
                verbose=True,
                print_freq=1)

    trainer.evaluate(test_dataloader)

    results['12class'][i+1] = trainer.prog

with open('results.json', 'w') as f:
    json.dump(results, f)