import os
import glob
import pandas as pd

from PIL import Image

normalglob = glob.glob('xray_data/0/*.jpg')
abnormalglob = glob.glob('xray_data/1/*.jpg')

trainglob = normalglob[:int(len(normalglob)*0.8)]
testglob = normalglob[int(len(normalglob)*0.8):]

trainglob2 = abnormalglob[:int(len(abnormalglob)*0.8)]
testglob2 = abnormalglob[int(len(abnormalglob)*0.8):]

train_path = []
train_label = []
test_path = []
test_label = []

for filepath in trainglob:
    img = Image.open(filepath)
    img = img.resize((224, 224))
    imgname = filepath.split('/')[-1]
    newpath = '/root/ktg/Capstone2/dataset/img/train/' + imgname
    img.save(newpath)
    train_path.append(newpath)
    train_label.append(0)

for filepath in testglob:
    img = Image.open(filepath)
    img = img.resize((224, 224))
    imgname = filepath.split('/')[-1]
    newpath = '/root/ktg/Capstone2/dataset/img/test/' + imgname
    img.save(newpath)
    test_path.append(newpath)
    test_label.append(0)

for filepath in trainglob2:
    img = Image.open(filepath)
    img = img.resize((224, 224))
    imgname = filepath.split('/')[-1]
    newpath = '/root/ktg/Capstone2/dataset/img/train/' + imgname
    img.save(newpath)
    train_path.append(newpath)
    train_label.append(1)

for filepath in testglob2:
    img = Image.open(filepath)
    img = img.resize((224, 224))
    imgname = filepath.split('/')[-1]
    newpath = '/root/ktg/Capstone2/dataset/img/test/' + imgname
    img.save(newpath)
    test_path.append(newpath)
    test_label.append(1)

train_df = {}
test_df = {}
train_df['filepath'] = train_path
train_df['label'] = train_label
test_df['filepath'] = test_path
test_df['label'] = test_label

train_df = pd.DataFrame(train_df)
test_df = pd.DataFrame(test_df)

train_df.to_csv('/root/ktg/Capstone2/dataset/csv/train.csv')
test_df.to_csv('/root/ktg/Capstone2/dataset/csv/test.csv')

