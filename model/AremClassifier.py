import os as os
from pathlib import Path
import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.linear_model import LogisticRegression , LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pickle
from sklearn.tree import DecisionTreeClassifier


class AremClassifier:
    def __init__(self,path_to_dataset):
        self.path_to_dataset = path_to_dataset

    def preprocessing(self):
        position_dataset = pd.DataFrame()
        position_dataset['Output'] = None
        start_index = 0
        end_index = None
        list_of_directories = []
        current_working_directories=self.path_to_dataset
        os.chdir(current_working_directories)


        for i in os.listdir():
            if Path(i).suffix == ".pdf":
                pass
        else:
            list_of_directories.append(i)
        


        for list_of_directories_ in list_of_directories:
            os.chdir(current_working_directories + "//" + list_of_directories_)
            for dataset_within_directories in os.listdir():
                try:
                    with open(dataset_within_directories, 'r') as file:
                        content = file.readlines()
                        column_name = list(content[4].split('\n')[0].split(': ')[1].split(','))
                        dataset = pd.read_csv(dataset_within_directories, comment='#', names=column_name)

                # Append rows to position_dataset
                        position_dataset = position_dataset.append(dataset, ignore_index=True)

                # Calculate the range of rows for the current dataset
                        end_index = len(position_dataset) - 1

                # Set the 'Output' column for the newly appended rows
                        position_dataset.loc[start_index:end_index, 'Output'] = list_of_directories_
                        start_index = end_index + 1  # Move the start_index to the next position

                except pd.errors.ParserError as e:
                    print(f"Error reading {dataset_within_directories}: {e}")
                    continue  # Skip to the next file if there's an error

# Now position_dataset should contain all concatenated data

