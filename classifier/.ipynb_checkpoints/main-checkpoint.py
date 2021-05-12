import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
from train import *

from LearningCurve import *
from predictions import *
from nih import *

import numpy as np
import pandas as pd
#---------------------- on q
path_image = "/home/alghas0c/CheXclusion/images/"



def train_validate_test_split(df, train_percent=.8, validate_percent=.1, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    train.to_csv(r'/home/alghas0c/CheXclusion/NIH/train.csv', index=False)
    test.to_csv(r'/home/alghas0c/CheXclusion/NIH/test.csv', index=False)
    validate.to_csv(r'/home/alghas0c/CheXclusion//NIH/val.csv', index=False)
    return train, validate, test


diseases = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
       'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
       'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
# age_decile = ['0-20', '20-40', '40-60', '60-80', '80-']
age_decile = ['40-60', '60-80', '20-40', '80-', '0-20']
gender = ['M', 'F']

def main():
    df = pd.read_csv('/home/alghas0c/CheXclusion/NIH/Data_Entry.csv')
    train_validate_test_split(df)
    train_df_path ="/home/alghas0c/CheXclusion/NIH/train.csv"
    test_df_path = "/home/alghas0c/CheXclusion/NIH/test.csv"
    val_df_path = "/home/alghas0c/CheXclusion/NIH/val.csv"
    MODE = "test"  # Select "train" or "test", "Resume", "plot", "Threhold", "plot15"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv(train_df_path)
    #train_df = train_df[train_df["Fibrosis"] == 1]
    train_df_size = len(train_df)
    print("Train_df path", train_df_size)

    test_df = pd.read_csv(test_df_path)
    #test_df = test_df[test_df["Fibrosis"] == 1]
    test_df_size = len(test_df)
    print("test_df path", test_df_size)

    val_df = pd.read_csv(val_df_path)
    #val_df = val_df[val_df["Fibrosis"] == 1]
    val_df_size = len(val_df)
    print("val_df path", val_df_size)

    if MODE == "train":
        ModelType = "densenet"  # select 'ResNet50','densenet','ResNet34', 'ResNet18'
        CriterionType = 'BCELoss'
        LR = 0.5e-3

        model, best_epoch = ModelTrain(train_df_path, val_df_path, path_image, ModelType, CriterionType, device,LR)

        PlotLearnignCurve()


    if MODE =="test":
        val_df = pd.read_csv(val_df_path)
        test_df = pd.read_csv(test_df_path)

        CheckPointData = torch.load('results/checkpoint')
        model = CheckPointData['model']

        make_pred_multilabel(model, test_df, val_df, path_image, device)


    if MODE == "Resume":
        ModelType = "Resume"  # select 'ResNet50','densenet','ResNet34', 'ResNet18'
        CriterionType = 'BCELoss'
        LR = 0.5e-3

        model, best_epoch = ModelTrain(train_df_path, val_df_path, path_image, result_path, ModelType, CriterionType, device,LR)

        PlotLearnignCurve()

    if MODE == "plot":
        gt = pd.read_csv("./results/True.csv")
        pred = pd.read_csv("./results/bipred.csv")
        factor = [gender, age_decile]
        factor_str = ['Patient Gender', 'Patient Age']
        for i in range(len(factor)):
            # plot_frequency(gt, diseases, factor[i], factor_str[i])
            # plot_TPR_NIH(pred, diseases, factor[i], factor_str[i])
            plot_sort_median(pred, diseases, factor[i], factor_str[i])
           # Actual_TPR(pred, diseases, factor[i], factor_str[i])

    #         plot_Median(pred, diseases, factor[i], factor_str[i])


if __name__ == "__main__":
    main()
