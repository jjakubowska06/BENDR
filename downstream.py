import torch
import tqdm
import argparse

import objgraph

import time
import utils
from result_tracking import ThinkerwiseResultTracker

from dn3.configuratron import ExperimentConfig
from dn3.data.dataset import Thinker
from dn3.trainable.processes import StandardClassification

from dn3_ext import BENDRClassification, LinearHeadBENDR

# Since we are doing a lot of loading, this is nice to suppress some tedious information
import mne
mne.set_log_level(False)

import warnings
warnings.simplefilter("ignore", DeprecationWarning)
import gc


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
    experiment = ExperimentConfig(args.ds_config)
    if args.results_filename:
        results = ThinkerwiseResultTracker()

    # loss_and_accuracy_file = open(args.logs_tracker, 'w')
    # set header for logs from training and validation
    # only for trainings now
    # ,Accuracy,loss,lr,epoch,iteration

    for ds_name, ds in tqdm.tqdm(experiment.datasets.items(), total=len(experiment.datasets.items()), desc='Datasets'):
        added_metrics, retain_best, _ = utils.get_ds_added_metrics(ds_name, args.metrics_config)
        iterator = tqdm.tqdm(utils.get_lmoso_iterator(ds_name, ds))
        for fold, (training, validation, test) in enumerate(iterator):
            tqdm.tqdm.write(torch.cuda.memory_summary())
            if args.model == utils.MODEL_CHOICES[0]:
                model = BENDRClassification.from_dataset(training, multi_gpu=args.multi_gpu)
            else:
                model = LinearHeadBENDR.from_dataset(training)

            if not args.random_init:
                model.load_pretrained_modules(experiment.encoder_weights, experiment.context_weights,
                                              freeze_encoder=args.freeze_encoder)
            process = StandardClassification(model, metrics=added_metrics)
            process.set_optimizer(torch.optim.Adam(process.parameters(), ds.lr, weight_decay=0.01))
            # AdamW, RAdam

            # Fit everything
            output_training, output_validation = process.fit(training_dataset=training, validation_dataset=validation, warmup_frac=0.1,
                        retain_best=retain_best, pin_memory=True, **ds.train_params)
            

            if args.results_filename:
                if isinstance(test, Thinker):
                    results.add_results_thinker(process, ds_name, test)
                else:
                    results.add_results_all_thinkers(process, ds_name, test, Fold=fold+1)
                results.to_spreadsheet(args.results_filename)
            
            if fold==1:
                # save model
                model.save(args.model_path)
                # save outputs from the first fold:
                output_training.to_csv(args.logs_tracker_train, mode ='a', header=None,index=False)
                output_validation.to_csv(args.logs_tracker_valid, mode ='a', header=None,index=False)
                # later we can do it without saving but for now it is what it is
            
            # free up RAM!!!
            del test
            del training, validation
            # explicitly garbage collect here, don't want to fit two models in GPU at once
            del process
            # objgraph.show_backrefs(model, filename='sample-backref-graph.png')
    
            del model
            torch.cuda.synchronize()
            time.sleep(10)
            gc.collect()
        
        # print("Creating ad saving plots...")
        # print("Plots saved.")
        print("done")
        
        if args.results_filename:
            results.performance_summary(ds_name)
            results.to_spreadsheet(args.results_filename)
