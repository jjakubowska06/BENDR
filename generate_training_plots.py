import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_best(df):
    '''
    Get best metrics (acc and loss) from each epoch.
    Input:
        df: DataFrame
    Output:
        DataFrame with one less dimention with best metric values from each epoch
    '''
    n_epochs = int(df['epoch'].iloc[-1])
    best_df = pd.DataFrame(columns=['acc','loss','lr','epoch','iteration'])

    for i in range(1, n_epochs+1):
        min_loss_value_id = df[df['epoch'] == i].loss.idxmin()
        #print(df.iloc[1])
        best_df.loc[i] = df.loc[min_loss_value_id]     
    return best_df


def plot_metric(df_train, df_valid, metric='Accuracy', save_file=''):
    plt.figure(figsize=[10,5])
    n_epochs = df_train.shape[0]
    if metric=="acc":
        ylim=[0,1.05]
    else:
        ylim=[0, 2.5] #arbitralnie

    plt.subplot(1,2,1) # metrics plot for training dataset
    plt.plot(df_train[metric])
    plt.xlabel('Epochs')
    plt.xlim(0, n_epochs)
    plt.ylabel(metric)
    plt.title("Training set")
    plt.ylim(ylim)

    plt.subplot(1,2,2) # metrics plot for validation dataset
    plt.plot(df_valid[metric])
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.title("Validation set")
    plt.ylim(ylim)


    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.suptitle(f"{metric} plots during training the model")
    plt.savefig(save_file)
    plt.close('all')


if __name__=="__main__":
    # docelowo bedziemy brac w locie z treningu a nie z plikow, nietrudne 

    model_desc = 'logs_AASM-hpf05_emg' 
    trainset_logs_path = "results/train-" + model_desc + '.csv' #train-linear-logs_radam_250e_60b-01wd.csv"
    validset_logs_path =  "results/valid-" + model_desc + '.csv'
    
    column_names=['acc','loss','lr','epoch','iteration']

    df_train = pd.read_csv(trainset_logs_path, names=column_names)
    df_valid = pd.read_csv(validset_logs_path, names=column_names)

    # get best scores of accuracy from training for each epoch
    best_df_train = get_best(df_train)
    # get best scores of accuracy from validation for each epoch
    # for valid it is different!
    # best_df_valid = get_best(df_valid)

    plot_metric(best_df_train, df_valid, 'acc', 'results/plots/acc-' + model_desc + '.png') # acc-linear-logs_radam_250e_60b-01wd.png')
    plot_metric(best_df_train, df_valid, 'loss', 'results/plots/loss-' + model_desc + '.png')

    # plt.subplot(1,2,1)
    # plt.plot(best_df_train['acc'])
    # plt.subplot(1,2,2)
    # plt.plot(df_valid['acc'])
    # plt.show()
    # plt.savefig('results/plots/acc-comparison.png')
    # plt.close('all')

    # plt.subplot(1,2,1)
    # plt.plot(best_df_train['loss'])
    # plt.subplot(1,2,2)
    # plt.plot(df_valid['loss'])
    # plt.savefig('results/plots/loss-comparison.png')


    # get best scores of loss from training for each epoch
    # best_loss_train = get_best()
    # get best scores of loss from validation for each epoch
    # best_loss_valid = get_best()

    # figure 1 -> accuracies for training and validation (epoch)
    # plt.subplot(1,2,1)  # -> accuracy from training
    # plot1: plot_accuracy(best_acc_train)
    # plt.subplot(1,2,2)  # -> accuracy from validation
    # plot2: plot_accuracy(best_acc_valid)
    # save to a file 

    # figure 2 -> losses for training and validation (epoch)
    # plt.subplot(1,2,1)  # -> accuracy from training
    # plot1: plot_accuracy(best_loss_train)
    # plt.subplot(1,2,2)  # -> accuracy from validation
    # plot1: plot_accuracy(best_loss_valid)
    # save to a file 




