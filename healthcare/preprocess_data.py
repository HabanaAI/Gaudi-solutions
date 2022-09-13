import pandas as pd
import csv
import os


def load_dataframe(csv_file):
    dataframe = pd.read_csv(csv_file)
    dataframe = dataframe[['Image Index', 'Finding Labels']]
    return dataframe


def convert_dataframe(dataframe):
    CLASSES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
               'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
    dataframe[CLASSES] = dataframe['Finding Labels'].apply(lambda labels: pd.Series([class_name in labels for class_name in CLASSES])).astype(int)
    dataframe = dataframe.drop(['Finding Labels'], axis=1)

    return dataframe


def filter_dataframe(dataframe, file_txt):
    with open(file_txt) as f:
        lines = f.readlines()
        lines = [s.rstrip() for s in lines]
        filtered_dataframe = dataframe[dataframe["Image Index"].isin(lines)]

    return filtered_dataframe


def save_dataframe(dataframe, output_file_txt):
    dataframe.to_csv(output_file_txt, header=False, index=False, sep=' ', quoting=csv.QUOTE_NONE)


def create_label_dir(path):
    if os.path.exists(path):
        raise NameError('The labels directory already exists.')
    os.mkdir(path)


def main():
    labels_file_csv = 'Data_Entry_2017_v2020.csv'
    train_txt_file  = 'train_val_list.txt'
    test_txt_file   = 'test_list.txt'
    labels_dir       = 'labels'
    try:
        create_label_dir(labels_dir)
    except NameError:
        print("NameError:: The labels directory already exists, check current directory.")
        exit(-1)
    dataframe = load_dataframe(labels_file_csv)
    dataframe = convert_dataframe(dataframe)

    train_dataframe = filter_dataframe(dataframe, train_txt_file)
    test_dataframe = filter_dataframe(dataframe, test_txt_file)

    save_dataframe(train_dataframe, labels_dir+'/train_list.txt')
    save_dataframe(test_dataframe, labels_dir+'/test_list.txt')


if __name__ == '__main__':
    main()
