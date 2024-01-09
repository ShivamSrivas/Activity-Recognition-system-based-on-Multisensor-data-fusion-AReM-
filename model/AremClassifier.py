import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score
from datetime import datetime
import pickle

class AremClassifier:
    def __init__(self, path_to_dataset):
        self.path_to_dataset = path_to_dataset

    def preprocessing(self):
        position_dataset = pd.DataFrame()
        position_dataset['Output'] = None
        start_index = 0
        end_index = None
        list_of_directories = []
        current_working_directories = self.path_to_dataset
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

        position_dataset.drop('time', axis=1, inplace=True)
        column_list = list(position_dataset.columns)
        column_list.pop(0)

        for column in column_list:
            position_dataset[column] = position_dataset[column].fillna(position_dataset[column].median())

        # Remove outliers based on quantile ranges
        quantile_ranges = {
            'avg_rss12': 0.95,
            'var_rss12': 0.95,
            'avg_rss13': 0.95,
            'var_rss13': 0.98,
            'avg_rss23': 0.95,
            'var_rss23': 0.95,
        }

        for feature, quantile_range in quantile_ranges.items():
            threshold = position_dataset[feature].quantile(quantile_range)
            position_dataset = position_dataset[position_dataset[feature] < threshold]

        # Split data into train and test sets
        x_train, x_test, y_train, y_test = train_test_split(
            position_dataset.drop("Output", axis=1),
            position_dataset['Output'],
            test_size=0.2,
            random_state=42
        )

        # Standardize features
        scaler = StandardScaler()
        x_standard_train = scaler.fit_transform(x_train)
        x_standard_train = pd.DataFrame(x_standard_train, columns=x_train.columns)

        # Calculate VIF
        vif_data = pd.DataFrame(
            [(x_standard_train.columns[i], variance_inflation_factor(x_standard_train.values, i))
             for i in range(x_standard_train.shape[1])],
            columns=["FEATURE", "VIF_SCORE"]
        )
        print(vif_data)

        return x_train, x_test, y_train, y_test

    def fit(self, x_train, y_train):
        decision_model = DecisionTreeClassifier()
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'ccp_alpha': [0.0, 0.1, 0.2, 0.5, 1.0]
        }
        grid_search = GridSearchCV(decision_model, param_grid, cv=5, scoring='accuracy')
        final_model = grid_search.fit(x_train, y_train)
        return final_model
    
    def modelSaved(self):
        x_train, x_test, y_train, y_test=self.preprocessing()
        model=self.fit(x_train,y_train)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        x_train_predict = model.predict(x_train)
        x_test_predict = model.predict(x_test)
        filename = r"C:\\Users\\hp\\Desktop\\Activity-Recognition-system-based-on-Multisensor-data-fusion-AReM-\\model\\savedModel\\saved_Model.pkl"
        with open(filename, 'wb') as file:
            pickle.dump(model, file)

        train_score,test_score=accuracy_score(x_train_predict,y_train),accuracy_score(x_test_predict,y_test)
        return f"Your Model is Saved and the train Score is {train_score} meanwhile your test accuracy is {test_score}"
        # Now position_dataset should contain all concatenated data

# Example usage:
# classifier = AremClassifier(path_to_dataset='your_path_to_dataset')
# x_train, y_train, x_test, y_test = classifier.preprocessing()
# classifier.fit(x_train, y_train)
