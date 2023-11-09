import pandas as pd
import numpy as np

def get_best(df, type='acc'):
    '''
    Get best metric from each epoch.
    Input:
        df: DataFrame
        type: string, 'acc' or 'loss'
    Output:
        DataFrame with one less dimention with best metric values from each epoch
    '''
    print(df[''])

    # mamy dataframe, wybieramy kolumne 0 lub 1 w zależnosci czy to acc czy loss
    # iterujemy sie po kazdej epoce i wybieramy najlepsze wartosci z iterations (ostatnia kolumna)
    # zapisujemy wektor najlepszych wartości acc/loss (epoch)
    return get_best

def plot_loss():

    return plot, xasis

def plot_accuracy():
    return plot, xasis

if __name__=="__main__":
    # docelowo bedziemy brac w locie z treningu a nie z plikow, nietrudne 

    trainset_logs_path = "results/logs-train.csv"
    validset_logs_path = "results/logs-valid.csv"
    
    column_names = ['nr','acc','loss','lr','epoch','iteration']
    df_train = pd.read_csv(trainset_logs_path, names=column_names)
    df_valid = pd.read_csv(validset_logs_pat, names=column_names)
    
    # get best scores of accuracy from training for each epoch
    # best_acc_train = get_best()
    # get best scores of accuracy from validation for each epoch
    # best_acc_valid = get_best()
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




