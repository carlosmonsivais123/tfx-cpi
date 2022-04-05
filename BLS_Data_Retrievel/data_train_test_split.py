import pandas as pd

class Data_Pre_Process:

    def sub_split_train_test(self, df):
        train_index = int(round(df.shape[0] * 0.8, 0))
        training_data = df.iloc[:train_index, :]

        testing_data = df.iloc[train_index: , :]

        return training_data, testing_data
