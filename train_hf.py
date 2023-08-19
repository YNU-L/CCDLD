import os
import random
import _pickle as pickle

import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np

from models.model import CCDLD
from metrics import EvaluateHFCallBack
from utils import DataGenerator


seed = 6669
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


mimic3_path = os.path.join('data', 'mimic3')
encoded_path = os.path.join(mimic3_path, 'encoded')
standard_path = os.path.join(mimic3_path, 'standard')
other_path = os.path.join(mimic3_path, 'other')


def load_data() -> (tuple, tuple, dict):
    code_map = pickle.load(open(os.path.join(encoded_path, 'code_map.pkl'), 'rb'))
    dictionary = pickle.load(open(os.path.join(encoded_path, 'dictionary.pkl'), 'rb'))

    code_category = pickle.load(open(os.path.join(encoded_path, 'code_category.pkl'), 'rb'))
    patient_dataset = pickle.load(open(os.path.join(standard_path, 'patient_dataset.pkl'), 'rb'))
    codes_dataset = pickle.load(open(os.path.join(standard_path, 'codes_dataset.pkl'), 'rb'))
    time_dataset = pickle.load(open(os.path.join(standard_path, 'time_dataset.pkl'), 'rb'))
    note_dataset = pickle.load(open(os.path.join(standard_path, 'note_dataset.pkl'), 'rb'))
    hf_dataset = pickle.load(open(os.path.join(standard_path, 'heart_failure.pkl'), 'rb'))
    auxiliary = pickle.load(open(os.path.join(standard_path, 'auxiliary.pkl'), 'rb'))
    return (code_map, dictionary, code_category), (patient_dataset, codes_dataset, time_dataset, note_dataset, hf_dataset), auxiliary


if __name__ == '__main__':
    (code_map, dictionary, code_category), (patient_dataset, codes_dataset, time_dataset, note_dataset, hf_dataset), auxiliary = load_data()
    train_patient_data, valid_patient_data, test_patient_data = patient_dataset['train_patient_data'], patient_dataset['valid_patient_data'], patient_dataset['test_patient_data']
    train_codes_data, valid_codes_data, test_codes_data = codes_dataset['train_codes_data'], codes_dataset['valid_codes_data'], codes_dataset['test_codes_data']
    train_time_data, valid_time_data, test_time_data = time_dataset['train_time_data'], time_dataset['valid_time_data'], time_dataset['test_time_data']
    train_note_data, valid_note_data, test_note_data = note_dataset['train_note_data'], note_dataset['valid_note_data'], note_dataset['test_note_data']
    train_hf_y, valid_hf_y, test_hf_y = hf_dataset['train_hf_y'], hf_dataset['valid_hf_y'], hf_dataset['test_hf_y']

    (train_patient_gender, train_patient_age, train_patient_cluster) = train_patient_data
    (valid_patient_gender, valid_patient_age, valid_patient_cluster) = valid_patient_data
    (test_patient_gender, test_patient_age, test_patient_cluster) = test_patient_data
    (train_codes_x, train_codes_y, train_visit_lens) = train_codes_data
    (valid_codes_x, valid_codes_y, valid_visit_lens) = valid_codes_data
    (test_codes_x, test_codes_y, test_visit_lens) = test_codes_data
    (train_time_x, train_time_y) = train_time_data
    (valid_time_x, valid_time_y) = valid_time_data
    (test_time_x, test_time_y) = test_time_data
    (train_note_x, train_note_lens, tf_idf_weight) = train_note_data
    (valid_note_x, valid_note_lens) = valid_note_data
    (test_note_x, test_note_lens) = test_note_data
    code_levels, patient_code_adj, code_code_adj = auxiliary['code_levels'], auxiliary['patient_code_adj'], auxiliary['code_code_adj']

    config = {
        'patient_code_adj': tf.constant(patient_code_adj, dtype=tf.float32),
        'code_code_adj': tf.constant(code_code_adj, dtype=tf.float32),
        'code_levels': tf.constant(code_levels, dtype=tf.int32),
        'code_category': tf.constant(code_category, dtype=tf.int32),
        'code_num_in_levels': np.max(code_levels, axis=0) + 1,
        'patient_num': train_codes_x.shape[0],
        'max_visit_seq_len': train_codes_x.shape[1],
        'max_note_seq_len': train_note_x.shape[1],
        'word_num': len(dictionary) + 1,
        'output_dim': 1,
        'category_num': code_category.max() + 1,
        'use_note': True,
        'use_attention': True,
        'use_patient_feature': True,
        'lambda': 0.1,
        'activation': 'sigmoid',
        'dropout': 0.2
    }

    visit_rnn_dims = [200]
    hyper_params = {
        'code_dims': [32, 32, 32, 32],
        'patient_dim': 16,
        'patient_gender_dim': 40,
        'patient_age_dim': 40,
        'patient_cluster_dim': 2,
        'word_dim': 16,
        'code_category_dim': 128,
        'patient_hidden_dims': [32],
        'code_hidden_dims': [64, 128],
        'visit_rnn_dims': visit_rnn_dims,
        'visit_attention_dim': 32,
        'soft_attention_dim': 400,
        'note_attention_dim': visit_rnn_dims[-1]
    }

    test_codes_gen = DataGenerator(
        [test_codes_x, test_visit_lens, test_note_x, test_note_lens, test_patient_gender, test_patient_age,
         test_patient_cluster], shuffle=False)

    def lr_schedule_fn(epoch, lr):
        if epoch < 4:
            lr = 0.001
        elif epoch < 20:
            lr = 0.001
        elif epoch < 50:
            lr = 0.0001
        else:
            lr = 0.0001
        return lr

    lr_scheduler = LearningRateScheduler(lr_schedule_fn)
    test_callback = EvaluateHFCallBack(test_codes_gen, test_hf_y)

    ccdld_model = CCDLD(config, hyper_params)
    ccdld_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=[tf.metrics.AUC()])
    ccdld_model.fit(x={
        'visit_codes': train_codes_x,
        'visit_lens': train_visit_lens,
        'word_ids': train_note_x,
        'word_lens': train_note_lens,
        'tf_idf_weight': tf_idf_weight,
        'patient_gender': train_patient_gender,
        'patient_age': train_patient_age,
        'patient_cluster': train_patient_cluster
    }, y=train_hf_y.astype(float), validation_data=({
        'visit_codes': valid_codes_x,
        'visit_lens': valid_visit_lens,
        'word_ids': valid_note_x,
        'word_lens': valid_note_lens,
        'patient_gender': valid_patient_gender,
        'patient_age': valid_patient_age,
        'patient_cluster': valid_patient_cluster
    }, valid_hf_y.astype(float)), epochs=8, batch_size=32, callbacks=[lr_scheduler, test_callback])
    ccdld_model.summary()
