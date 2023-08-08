import os
import _pickle as pickle
import numpy as np

from preprocess.parse_csv4 import parse_admission, parse_diagnoses, parse_patients, parse_notes
from preprocess.parse_csv4 import calibrate_patient_by_admission, calibrate_patient_by_notes
from preprocess.encode import encode_code, encode_time_duration, encode_note_train, encode_note_test, encode_category
from preprocess.build_dataset import split_patients4, build_patient_data, build_disease_y, sample_patients
from preprocess.build_dataset import build_code_xy, build_time_duration_xy, build_note_x, build_tf_idf_weight
from preprocess.auxiliary import generate_code_levels, generate_patient_code_adjacent, generate_code_code_adjacent, co_occur


if __name__ == '__main__':
    data_path = 'data'
    max_note_len = 50000
    raw_path = os.path.join(data_path, 'mimic4', 'raw')
    other_path = os.path.join(data_path, 'mimic4', 'other')
    if not os.path.exists(raw_path):
        os.makedirs(raw_path)
        print('please put the CSV files in `data/mimic4/raw`')
        exit()
    if not os.path.exists(other_path):
        os.makedirs(other_path)
    patient_admission = parse_admission(raw_path)
    print('There are %d patients' % len(patient_admission))
    admission_codes = parse_diagnoses(raw_path, patient_admission)
    calibrate_patient_by_admission(patient_admission, admission_codes)
    print('There are %d patients' % len(patient_admission))

    patient_note = parse_notes(raw_path, patient_admission)
    calibrate_patient_by_notes(patient_admission, admission_codes, patient_note)
    print('There are %d patients' % len(patient_admission))
    patient_admission, admission_codes = sample_patients(patient_admission, admission_codes, 10000)
    print('There are %d patients' % len(patient_admission))
    patient_gender, patient_age, patient_cluster = parse_patients(raw_path, patient_admission, admission_codes)

    max_admission_num = 0
    for pid, admissions in patient_admission.items():
        if len(admissions) > max_admission_num:
            max_admission_num = len(admissions)
    print('max_admission_num is %d' % max_admission_num)
    max_code_num_in_a_visit = 0
    for admission_id, codes in admission_codes.items():
        if len(codes) > max_code_num_in_a_visit:
            max_code_num_in_a_visit = len(codes)
    print('max_code_num_in_a_visit %d' % max_code_num_in_a_visit)

    admission_codes_encoded, code_map = encode_code(admission_codes)
    patient_time_duration_encoded = encode_time_duration(patient_admission)
    code_category = encode_category(data_path, code_map)
    code_num = len(code_map)
    print("code_num: ", code_num)

    train_pids, valid_pids, test_pids = split_patients4(
        patient_admission=patient_admission,
        admission_codes=admission_codes,
        code_map=code_map
    )

    train_note_encoded, dictionary = encode_note_train(patient_note, train_pids, max_note_len=max_note_len)
    valid_note_encoded = encode_note_test(patient_note, valid_pids, dictionary, max_note_len=max_note_len)
    test_note_encoded = encode_note_test(patient_note, test_pids, dictionary, max_note_len=max_note_len)

    def max_word_num(note_encoded: dict) -> int:
        return max(len(note) for note in note_encoded.values())

    max_word_num_in_a_note = max([max_word_num(train_note_encoded), max_word_num(valid_note_encoded), max_word_num(test_note_encoded)])
    print('max word num in a note:', max_word_num_in_a_note)

    train_patient_gender, train_patient_age, train_patient_cluster = build_patient_data(train_pids, patient_gender,
                                                                                        patient_age, patient_cluster)
    valid_patient_gender, valid_patient_age, valid_patient_cluster = build_patient_data(valid_pids, patient_gender,
                                                                                        patient_age, patient_cluster)
    test_patient_gender, test_patient_age, test_patient_cluster = build_patient_data(test_pids, patient_gender,
                                                                                        patient_age, patient_cluster)

    train_codes_x, train_codes_y, train_visit_lens = build_code_xy(train_pids, patient_admission, admission_codes_encoded, max_admission_num, code_num, max_code_num_in_a_visit)
    valid_codes_x, valid_codes_y, valid_visit_lens = build_code_xy(valid_pids, patient_admission, admission_codes_encoded, max_admission_num, code_num, max_code_num_in_a_visit)
    test_codes_x, test_codes_y, test_visit_lens = build_code_xy(test_pids, patient_admission, admission_codes_encoded, max_admission_num, code_num, max_code_num_in_a_visit)

    train_time_x, train_time_y = build_time_duration_xy(train_pids, patient_time_duration_encoded, max_admission_num)
    valid_time_x, valid_time_y = build_time_duration_xy(valid_pids, patient_time_duration_encoded, max_admission_num)
    test_time_x, test_time_y = build_time_duration_xy(test_pids, patient_time_duration_encoded, max_admission_num)

    train_note_x, train_note_lens = build_note_x(train_pids, train_note_encoded, max_word_num_in_a_note)
    valid_note_x, valid_note_lens = build_note_x(valid_pids, valid_note_encoded, max_word_num_in_a_note)
    test_note_x, test_note_lens = build_note_x(test_pids, test_note_encoded, max_word_num_in_a_note)
    tf_idf_weight = build_tf_idf_weight(train_pids, train_note_x, train_note_encoded, len(dictionary))

    train_hf_y = build_disease_y('428', train_codes_y, code_map)
    valid_hf_y = build_disease_y('428', valid_codes_y, code_map)
    test_hf_y = build_disease_y('428', test_codes_y, code_map)

    code_levels = generate_code_levels(data_path, code_map)
    patient_code_adj = generate_patient_code_adjacent(code_x=train_codes_x, code_num=code_num)
    code_code_adj_t = generate_code_code_adjacent(code_level_matrix=code_levels, code_num=code_num)
    co_occur_matrix = co_occur(train_pids, patient_admission, admission_codes_encoded, code_num)
    code_code_adj = code_code_adj_t * co_occur_matrix

    train_codes_data = (train_codes_x, train_codes_y, train_visit_lens)
    valid_codes_data = (valid_codes_x, valid_codes_y, valid_visit_lens)
    test_codes_data = (test_codes_x, test_codes_y, test_visit_lens)
    train_time_data = (train_time_x, train_time_y)
    valid_time_data = (valid_time_x, valid_time_y)
    test_time_data = (test_time_x, test_time_y)
    train_note_data = (train_note_x, train_note_lens, tf_idf_weight)
    valid_note_data = (valid_note_x, valid_note_lens)
    test_note_data = (test_note_x, test_note_lens)
    train_patient_data = (train_patient_gender, train_patient_age, train_patient_cluster)
    valid_patient_data = (valid_patient_gender, valid_patient_age, valid_patient_cluster)
    test_patient_data = (test_patient_gender, test_patient_age, test_patient_cluster)

    l1 = len(train_pids)
    train_patient_ids = np.arange(0, l1)
    l2 = l1 + len(valid_pids)
    valid_patient_ids = np.arange(l1, l2)
    l3 = l2 + len(test_pids)
    test_patient_ids = np.arange(l2, l3)
    pid_map = dict()
    for i, pid in enumerate(train_pids):
        pid_map[pid] = train_patient_ids[i]
    for i, pid in enumerate(valid_pids):
        pid_map[pid] = valid_patient_ids[i]
    for i, pid in enumerate(test_pids):
        pid_map[pid] = test_patient_ids[i]

    mimic4_path = os.path.join('data', 'mimic4')
    encoded_path = os.path.join(mimic4_path, 'encoded')
    if not os.path.exists(encoded_path):
        os.makedirs(encoded_path)
    print('saving encoded data ...')
    pickle.dump(patient_admission, open(os.path.join(encoded_path, 'patient_admission.pkl'), 'wb'))
    pickle.dump(admission_codes_encoded, open(os.path.join(encoded_path, 'codes_encoded.pkl'), 'wb'))
    pickle.dump(patient_time_duration_encoded, open(os.path.join(encoded_path, 'time_encoded.pkl'), 'wb'))
    pickle.dump({
        'train_note_encoded': train_note_encoded,
        'valid_note_encoded': valid_note_encoded,
        'test_note_encoded': test_note_encoded
    }, open(os.path.join(encoded_path, 'note_encoded.pkl'), 'wb'))
    pickle.dump(code_map, open(os.path.join(encoded_path, 'code_map.pkl'), 'wb'))
    pickle.dump(code_category, open(os.path.join(encoded_path, 'code_category.pkl'), 'wb'))
    pickle.dump(dictionary, open(os.path.join(encoded_path, 'dictionary.pkl'), 'wb'))
    pickle.dump(pid_map, open(os.path.join(encoded_path, 'pid_map.pkl'), 'wb'))

    print('saving standard data ...')
    standard_path = os.path.join(mimic4_path, 'standard')
    if not os.path.exists(standard_path):
        os.makedirs(standard_path)
    pickle.dump({
        'train_pids': train_pids,
        'valid_pids': valid_pids,
        'test_pids': test_pids
    }, open(os.path.join(standard_path, 'patient_pids.pkl'), 'wb'))
    pickle.dump({
        'train_patient_data': train_patient_data,
        'valid_patient_data': valid_patient_data,
        'test_patient_data': test_patient_data
    }, open(os.path.join(standard_path, 'patient_dataset.pkl'), 'wb'))
    pickle.dump({
        'train_codes_data': train_codes_data,
        'valid_codes_data': valid_codes_data,
        'test_codes_data': test_codes_data
    }, open(os.path.join(standard_path, 'codes_dataset.pkl'), 'wb'))
    pickle.dump({
        'train_time_data': train_time_data,
        'valid_time_data': valid_time_data,
        'test_time_data': test_time_data
    }, open(os.path.join(standard_path, 'time_dataset.pkl'), 'wb'))
    pickle.dump({
        'train_note_data': train_note_data,
        'valid_note_data': valid_note_data,
        'test_note_data': test_note_data
    }, open(os.path.join(standard_path, 'note_dataset.pkl'), 'wb'))
    pickle.dump({
        'train_hf_y': train_hf_y,
        'valid_hf_y': valid_hf_y,
        'test_hf_y': test_hf_y
    }, open(os.path.join(standard_path, 'heart_failure.pkl'), 'wb'))
    pickle.dump({
        'code_levels': code_levels,
        'patient_code_adj': patient_code_adj,
        'code_code_adj': code_code_adj,
    },  open(os.path.join(standard_path, 'auxiliary.pkl'), 'wb'))
