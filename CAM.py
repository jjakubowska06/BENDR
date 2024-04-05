import tqdm
import pandas as pd
import numpy as np
import os
from pathlib import Path


import torch
import argparse

import objgraph

import time
import utils

from dn3.configuratron import ExperimentConfig
from dn3.data.dataset import Thinker
from dn3.trainable.processes import StandardClassification
from result_tracking import ThinkerwiseResultTracker

from dn3.transforms.instance import To1020

# for plotting the model
# from torchviz import make_dot
# from torch.utils.tensorboard import SummaryWriter
# import tensorflow.contrib.slim.nets as nets
# import hiddenlayer as hl
# import hiddenlayer.transforms as ht

from torchvision import models, transforms


from dn3_ext import BENDRClassification, LinearHeadBENDR
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tunes BENDER models.")
    parser.add_argument('model', choices=utils.MODEL_CHOICES)
    parser.add_argument('--ds-config', default="configs/downstream.yml", help="The DN3 config file to use.")
    parser.add_argument('--metrics-config', default="configs/metrics.yml", help="Where the listings for config "
                                                                                "metrics are stored.")
    parser.add_argument('--subject-specific', action='store_true', help="Fine-tune on target subject alone.")
    parser.add_argument('--mdl', action='store_true', help="Fine-tune on target subject using all extra data.")
    parser.add_argument('--freeze-encoder', action='store_true', help="Whether to keep the encoder stage frozen. "
                                                                      "Will only be done if not randomly initialized.")
    parser.add_argument('--random-init', action='store_true', help='Randomly initialized BENDR for comparison.')
    parser.add_argument('--multi-gpu', action='store_true', help='Distribute BENDR over multiple GPUs')
    parser.add_argument('--num-workers', default=1, type=int, help='Number of dataloader workers.')
    parser.add_argument('--results-filename', default=None, help='What to name the spreadsheet produced with all '
                                                                 'final results.')
    parser.add_argument('--model-path', default="weights/model.pt", help='A path to save a model weights')
    parser.add_argument('--logs-tracker-train', default="results/logs-train.csv", help='A path to save logs from training')
    parser.add_argument('--logs-tracker-valid', default="results/logs-valid.csv", help='A path to save logs from validation')


args = parser.parse_args()

predictions_path = 'predictions/'
experiment_name = 'BENDR-polid-AASM-eeg1_emg.pt'

experiment = ExperimentConfig("configs/testing.yml")
test_subjects = ['s02'] #, 's06', 's18', 's23', 's25', 's29']

database_path = experiment.datasets['polid'].toplevel

# if not os.path.exists(os.path.join(predictions_path, experiment_name)):
#     os.mkdir(os.path.join(predictions_path, experiment_name))


for ds_name, ds_config in experiment.datasets.items():
    predictions = []
    for sub in test_subjects:
        ds_config.toplevel = Path(os.path.join(database_path, sub))
        print(ds_config.toplevel)

        added_metrics, retain_best, _ = utils.get_ds_added_metrics(ds_name, args.metrics_config)

        dataset = ds_config.auto_construct_dataset()
        
        # if jakis parametr z coonfiga: nie dodawaj lub dodaj 10-20
        dataset.add_transform(To1020())
        print(dataset.channels)

        per = dataset.thinkers['s02'].to_numpy()
        # print(per.shape)    # (1356,90,2560) = ilosc skladek, ilosc kanalow, ilosc probek w skladce ( t)

    
        if args.model == utils.MODEL_CHOICES[0]:
            model = BENDRClassification.from_dataset(dataset, multi_gpu=args.multi_gpu)
        else:
            model = LinearHeadBENDR.from_dataset(dataset)

        # model = LinearHeadBENDR.from_dataset(dataset)
        model.load(experiment.encoder_weights, include_classifier=True)


        # CAM - hook
        features_blobs = []
        
        def hook_feature(module, input, output):
            features_blobs.append(output.data.cpu().numpy())

        model._modules.get('enc_augment').register_forward_hook(hook_feature)
        params = list(model.parameters())
        weight_softmax = np.squeeze(params[-2].data.numpy())    # cyz na pewno -2?

        
        def returnCAM(feature_conv, weight_softmax, class_idx):
            size_upsample = (256, 256)
            # bz, 
            nc, h, w = feature_conv.shape
            output_cam = []
            for idx in class_idx:
                cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
                cam = cam.reshape(h, w)
                cam = cam - np.min(cam)
                cam_img = cam / np.max(cam)
                cam_img = np.uint8(255 * cam_img)
                output_cam.append(cv2.resize(cam_img, size_upsample))
            return output_cam

        preprocess = transforms.Compose([
            transforms.Resize((20, 2560)),  # size?
            transforms.ToTensor()])#,
            # normalize])


        print(model)
        

        # # make softmax predictions
        model = model.train(False)
        process = StandardClassification(model, metrics=added_metrics)

        x , y =  map(torch.Tensor, dataset.to_numpy())

        # # predictions
        inputs, outputs = process.predict(dataset)

        y_pred = outputs.softmax(dim=1)
        y_numpy = y_pred.numpy()

        # get tags with highest probability
        tags_BENDR = np.argmax(y_numpy, axis=1)
        true_tags = inputs[1].numpy()

        print(true_tags.shape, tags_BENDR.shape)
        print(y_numpy)


        # sortujemy wyniki 
        # probs, idx = y_numpy.sort(0, True)
        # probs = probs.numpy()
        # idx = idx.numpy()

        # for i in range(0, 5):
        #     print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))


        # CAM
        CAMs = returnCAM(features_blobs[0], weight_softmax, 0) # 3 arguemnt - klasa
        print(type(CAM))
        print(CAM)

        # print(f'output CAM.jpg for the top1 prediction: {0}')
        # img = cv2.imread(folder + 'test.jpg')
        # height, width, _ = img.shape
        # heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
        # result = heatmap * 0.3 + img * 0.5
        # cv2.imwrite(folder + 'CAM.jpg', result)
        # cv2_imshow(result) 
        
        # Colab i cv2.imshow() nie współpracują, więc musieliśmy zimportować patcha z cv2_imshow
        # print('output CAM.jpg for the top1 prediction: %s'%classes[idx[0]])
        # img = cv2.imread(folder + 'test.jpg')
        # height, width, _ = img.shape
        # heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
        # result = heatmap * 0.3 + img * 0.5
        # cv2.imwrite(folder + 'CAM.jpg', result)
        # cv2_imshow(result) # Colab i cv2.imshow() nie współpracują, więc musieliśmy zimportować patcha z cv2_imshow




        # in_and_out = np.stack((true_tags, tags_BENDR))
        # zapisac dla kazdego oddzielnie?

        # file_name = os.path.join(predictions_path, experiment_name, experiment_name + '_' + sub + '.npy')
        # file_no_softmax = os.path.join(predictions_path, experiment_name, experiment_name + '_' + sub + '_softmax.npy')
     
        # np.save(file_name, in_and_out)
        # np.save(file_no_softmax, y_numpy)

        # metrics = process.evaluate(dataset)
        # print(metrics)