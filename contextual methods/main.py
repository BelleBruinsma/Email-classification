import pandas as pd
import numpy as np

from multiprocessing import Process, freeze_support
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics
import shutil

from args import *
from classification_model import ClassificationModel


## initialize parameters
## change learning rate in args
# learning_rate = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
# args['learning_rate'] = learning_rate
## change epochs in args
# epoch = [1, 5, 10, 15, 20]
# args['num_train_epochs'] = epoch
## unfreezing encoding layer:
# unfr = 12


if __name__ == '__main__':

    freeze_support()

    # import data
    df1 = pd.read_csv("../data/", header=None, sep=';')

    df = df1.sample(frac=0.1, random_state=42)
    df.reset_index(drop=True, inplace=True)
    df.columns = ['text', 'label']

    ss = ShuffleSplit(n_splits=1, test_size=0.20)
    list_confs = []

    for train_, test_ in ss.split(df):
        # shuffle the train test split
        train = df.loc[train_]
        test = df.loc[test_]

        # reset columns
        train.columns = ['text', 'label']
        test.columns = ['text', 'label']

        # initialize model
        model = ClassificationModel(args['model_type'], args['model_name'], num_labels=3, args=args, use_cuda=False)

        # train model
        unfr=1
        model.train_model(train, unfreeze=unfr, show_running_loss=True, eval_df=test)
        model = ClassificationModel(args['model_type'], 'outputs/', use_cuda=False)

        # evaluate model
        result, model_outputs, wrong_predictions = model.eval_model(test)

        y_pred, y_test = np.argmax(model_outputs, axis=1).tolist(), test['label'].tolist()
        conf_mat = metrics.confusion_matrix(y_test, y_pred)

        # remove folders to prevent overwriting
        try:
            shutil.rmtree('./cache')
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

        try:
            shutil.rmtree('./cache_dir')
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

        try:
            shutil.rmtree('./outputs')
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))




