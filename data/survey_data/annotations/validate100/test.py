import json
import numpy as np
import os


with open('person_keypoints_validate_infant.json') as f1:
    train = json.load(f1)

print(len(train['images']))
print(len(train['annotations']))

num_img = len(train['images'])
supine = 0
prone = 0
sitting = 0
standing = 0
for i in range(num_img):
    if train['images'][i]['posture'] == 'supine':
        train['images'][i]['posture'] == 'Supine'
        supine += 1
    if train['images'][i]['posture'] == 'prone':
        train['images'][i]['posture'] == 'Prone'
        prone += 1
    if train['images'][i]['posture'] == 'sitting':
        train['images'][i]['posture'] == 'Sitting'
        sitting += 1
    if train['images'][i]['posture'] == 'standing':
        train['images'][i]['posture'] == 'Standing'
        standing += 1

with open('person_keypoints_validate_infant.json', 'w') as fb:
    json.dump(train, fb)

print(supine)
print(prone)
print(sitting)
print(standing)
        
