#  Copyright 2022 Tamanna, Licensed under MIT. For more information , check LICENSE.txt
import pandas
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

issue_df=pandas.DataFrame()

def loadDataset():
    global  issue_df
    try:
        # reading data from file
        issue_df= pd.read_csv('../data/dataInfoTypes.csv')
    except OSError as e:
        print('No data file has been given')

#This  function will provide necessary info about the dataset
def getDatasetInfo():
    #printing the number of labels and samples along with the class distribution
    print('Total ' + str(len(issue_df['Code'].unique())) + ' Classes and  ' +str(len(issue_df)) + ' samples in this dataset')
    print('\n--------------Class Distribution----------')
    print(issue_df.Code.value_counts())

    #printing dataset column information
    print('\nColumns in the dataset includes issue comment text and corresponding features. Columns are:')
    print(set(issue_df.head()))
    print('\n')
    return issue_df

def encodeLabels():
    #encoding the class name to numberical values and storing the original code in another column, 'Code (Original)'
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(issue_df['Code'])
    issue_df['Code (Original)'] = issue_df['Code']
    issue_df['Code'] = label_encoder.transform(issue_df['Code'])

    #mapping the encoded numerical values to the class names and prepared a dictionary, class_id_map
    class_id_map_df = issue_df.drop_duplicates(subset=['Code'])
    encoded_label_list = class_id_map_df['Code'].to_list()
    original_label_list = class_id_map_df['Code (Original)'].to_list()
    class_id_map = dict(zip(encoded_label_list, original_label_list))
    return class_id_map

def encodeCategAndBoolColumns():
    #Since these columns contain categorical and boolean values; transforming them into numerical values
    categ = ['aa', 'begauth', 'has_code', 'first_turn', 'last_turn']
    le = preprocessing.LabelEncoder()

    #test
    if (set(categ).issubset(set(issue_df.head()))==False):
        return issue_df

    issue_df[categ] = issue_df[categ].apply(le.fit_transform)
    return  issue_df

def trainModel(train_df):
    #dropping Text data, labels (Code', 'Text Content', 'Code (Original)') and Document colums since they will not be used for training
    X_train = train_df.drop(['Code', 'Text Content', 'Code (Original)','Document'], axis=1)
    y_train = train_df['Code'].values

    #training using RandomForestClassifier
    clf = RandomForestClassifier()

    #test
    try:
        clf.fit(X_train, y_train)
        print('Training Sucessfully completed')
    except ValueError as e:
        print('Training failed')

    return  clf

def testModel(test_df,clf,class_id_map):
    # dropping Text data, labels (Code', 'Text Content', 'Code (Original)') and Document colums
    X_test = test_df.drop(['Code', 'Text Content', 'Code (Original)','Document'], axis=1)
    y_test = test_df['Code'].values

    #test
    try:
        # predicting label
        y_pred = clf.predict(X_test)
        print('Testing Sucessfully completed')
    except ValueError as e:
        print('Testing failed')
        return



    #using the class_id_map to decode the class names for showing result in a easily understanding format
    true_labels = []
    pred_labels = []

    #test
    try:
        for e in y_test:
            true_labels.append(class_id_map[e])
        for e in y_pred:
            pred_labels.append(class_id_map[e])

        print('\n------------------------------Test Result----------------------------------')
        print(classification_report(true_labels, pred_labels))
    except KeyError as e:
        print('Error in genetating Classification Report')




def main():

    #This function loads dataset on issue_df
    loadDataset()

    #if df not loaded, return from here
    # unit_test
    if(len(issue_df)==0):
        return

    #This function gives insights into the dataset
    getDatasetInfo()

    #class labels and columns that contain categorical and boolean values(others contain numerical) are encoded to numerical values
    class_id_map=encodeLabels()
    encodeCategAndBoolColumns()

    # 90% are data are taken into training set and 10% into test set using stratify
    train_df, test_df = train_test_split(issue_df, test_size=0.1, random_state=10, stratify=issue_df['Code'])

    #training
    clf=trainModel(train_df)
    
    #testing
    testModel(test_df,clf,class_id_map)


# Code runs from here
main()



