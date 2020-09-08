import os
import pandas as pd
import numpy as np

def load_data(data_path):
    train_id_label = pd.read_csv(os.path.join(data_path, 'train_id_label.csv'), delimiter=',')
    train_images = np.load(os.path.join(data_path, 'train_images.npy'))
    
    valid_id_label = pd.read_csv(os.path.join(data_path, 'valid_id_label.csv'), delimiter=',')
    valid_images = np.load(os.path.join(data_path, 'valid_images.npy'))

    test_id = pd.read_csv(os.path.join(data_path, 'test_id.csv'), delimiter=',')
    test_images = np.load(os.path.join(data_path, 'test_images.npy'))
    
    return (train_id_label, train_images), (valid_id_label, valid_images), (test_id, test_images)


def save_prediction(pred):
    sub_df = pd.DataFrame()
    sub_df['Id'] = pred['id']
    sub_df['label'] = pred['label']

    sub_df.to_csv('submission.csv', index=None)

    print('\nSave Completed...!!')


def run(train_and_predict, train_data, valid_data, test_images):
    pred = train_and_predict(train_data, valid_data, test_images)
    
    save_prediction(pred)