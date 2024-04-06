import os
import csv
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def parse_admission(path) -> dict:
    print('parsing ADMISSIONS.csv ...')
    admission_path = os.path.join(path, 'ADMISSIONS.csv')
    admissions = pd.read_csv(
        admission_path,
        usecols=['SUBJECT_ID', 'HADM_ID', 'ADMITTIME'],
        converters={'SUBJECT_ID': np.int, 'HADM_ID': np.int, 'ADMITTIME': np.str}
    )
    all_patients = dict()
    for i, row in admissions.iterrows():
        if i % 100 == 0:
            print('\r\t%d in %d rows' % (i + 1, len(admissions)), end='')
        pid = row['SUBJECT_ID']
        admission_id = row['HADM_ID']
        admission_time = datetime.strptime(row['ADMITTIME'], '%Y-%m-%d %H:%M:%S')
        if pid not in all_patients:
            all_patients[pid] = []
        admission = all_patients[pid]
        admission.append({
            'admission_id': admission_id,
            'admission_time': admission_time
        })
    print('\r\t%d in %d rows' % (len(admissions), len(admissions)))
    patient_admission = dict()
    for pid, admissions in all_patients.items():
        if len(admissions) > 1:
            patient_admission[pid] = sorted(admissions, key=lambda admission: admission['admission_time'])
    return patient_admission


def parse_diagnoses(path, patient_admission: dict) -> dict:
    print('parsing DIAGNOSES_ICD.csv ...')
    diagnoses_path = os.path.join(path, 'DIAGNOSES_ICD.csv')
    diagnoses = pd.read_csv(
        diagnoses_path,
        usecols=['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE'],
        converters={'SUBJECT_ID': np.int, 'HADM_ID': np.int, 'ICD9_CODE': np.str}
    )

    def to_standard_icd9(code: str):
        split_pos = 4 if code.startswith('E') else 3
        icd9_code = code[:split_pos] + '.' + code[split_pos:] if len(code) > split_pos else code
        return icd9_code

    admission_codes = dict()
    for i, row in diagnoses.iterrows():
        if i % 100 == 0:
            print('\r\t%d in %d rows' % (i + 1, len(diagnoses)), end='')
        pid = row['SUBJECT_ID']
        if pid in patient_admission:
            admission_id = row['HADM_ID']
            code = row['ICD9_CODE']
            if code == '':
                continue
            code = to_standard_icd9(code)
            if admission_id not in admission_codes:
                codes = []
                admission_codes[admission_id] = codes
            else:
                codes = admission_codes[admission_id]
            codes.append(code)
    print('\r\t%d in %d rows' % (len(diagnoses), len(diagnoses)))
    return admission_codes


def process_patients(path, patient_admission: dict, admission_codes: dict):
    patients_path = os.path.join(path, 'PATIENTS.csv')
    patients = pd.read_csv(
        patients_path,
        usecols=['SUBJECT_ID', 'GENDER', 'DOB'],
        converters={'SUBJECT_ID': np.int, 'GENDER': np.str, 'DOB': np.str},
        low_memory=False)
    pids = list(patient_admission.keys())

    fil_patients = patients.loc[patients['SUBJECT_ID'].isin(pids)]
    data = patient_admission
    data2 = admission_codes
    dict_temp = {}
    dict_pid2diagnose = {}

    for i in data:
        list_a = data[i]
        new_list = list_a[:-1]
        if i not in dict_temp:
            dict_temp[i] = []
        for j in new_list:
            dict_temp[i].append(j['admission_id'])
    count = 0
    for i in dict_temp:
        if i not in dict_pid2diagnose:
            dict_pid2diagnose[i] = []
        try:
            for j in dict_temp[i]:
                dict_pid2diagnose[i].extend(data2[j])
        except:
            count += 1
    data = dict_pid2diagnose

    category = open('data/cate.txt', 'r')
    category_dict = {}
    dia_dict = {}
    count_dia = 1
    for line in category.readlines():
        split_line = line.split(':')
        key = split_line[0]
        value = split_line[1].replace('\n', '')
        if value not in dia_dict:
            dia_dict[value] = count_dia
            count_dia += 1
        if key not in category_dict:
            category_dict[key] = dia_dict[value]

    with open('data/mimic3/other/patients_temp.csv', mode='w', newline='') as temp_file:
        employee_writer = csv.writer(temp_file)
        header_list = ['user_id']
        for i in range(1, 158):
            header_list.append(i)
        employee_writer.writerow(header_list)
        for user_id in data:
            temp_list = [0 for _ in range(0, 158)]
            temp_list[0] = user_id
            for code in data[user_id]:
                i = code
                temp = i.split('.')[0]
                temp_list[category_dict[temp]] += 1
            employee_writer.writerow(temp_list)
    data_pre = pd.read_csv('data/mimic3/other/patients_temp.csv', usecols=[i for i in range(1, 158)], low_memory=False)
    data_pre2 = pd.read_csv('data/mimic3/other/patients_temp.csv', names=['user_id'], low_memory=False)

    pca = PCA(n_components=5)
    data = pca.fit_transform(data_pre)
    estimator = KMeans(n_clusters=10)
    estimator.fit(data)
    predict = estimator.predict(data)
    clusters = {}
    for index, i in enumerate(data_pre2.iterrows()):
        if index == 0:
            continue
        if int(i[0][0]) not in clusters:
            clusters[int(i[0][0])] = 0
    for index, item in enumerate(clusters.keys()):
        clusters[int(item)] = predict[index]

    fil_patients = fil_patients.assign(age=pd.Series(dtype=int))
    fil_patients = fil_patients.assign(cluster=pd.Series(dtype=int))
    new_data = patient_admission
    for i, row in fil_patients.iterrows():
        pid, DOB = row['SUBJECT_ID'], row['DOB']
        DOB = datetime.strptime(DOB, "%Y-%m-%d %H:%M:%S")
        age = int(abs(new_data[pid][-1]['admission_time'].year - DOB.year))
        cluster = clusters[pid]
        if age <= 9:
            age = 0
        elif 10 <= age <= 19:
            age = 1
        elif 20 <= age <= 29:
            age = 2
        elif 30 <= age <= 39:
            age = 3
        elif 40 <= age <= 49:
            age = 4
        elif 50 <= age <= 59:
            age = 5
        elif 60 <= age <= 69:
            age = 6
        elif 70 <= age <= 79:
            age = 7
        elif 80 <= age <= 89:
            age = 8
        elif 90 <= age:
            age = 9
        fil_patients.loc[i, 'age'] = int(age)
        fil_patients.loc[i, 'cluster'] = int(cluster)
    fil_patients.rename(columns={'SUBJECT_ID': 'subject_id', 'GENDER': 'gender'}, inplace=True)
    fil_patients[['subject_id', 'gender', 'age', 'cluster']].to_csv("data/mimic3/raw/PATIENTS_2.csv", index=False)


def parse_patients(path, patient_admission: dict, admission_codes: dict) -> (dict, dict, dict):
    print('parsing PATIENTS.csv ...')
    process_patients(path, patient_admission, admission_codes)
    print('parsing PATIENTS_2.csv ...')
    patients_path = os.path.join(path, 'PATIENTS_2.csv')
    patients = pd.read_csv(
        patients_path,
        usecols=['subject_id', 'gender', 'age', 'cluster'],
        converters={'subject_id': np.int, 'gender': np.str, 'age': np.float, 'cluster': np.float},
        low_memory=False)

    def gender_encode(gender: str):
        if gender == 'F':
            return 0
        if gender == 'M':
            return 1

    patient_gender = dict()
    patient_age = dict()
    patient_cluster = dict()
    for i, row in patients.iterrows():
        if i % 100 == 0:
            print('\r\t%d in %d rows' % (i + 1, len(patients)), end='')
        pid = row['subject_id']
        if pid in patient_admission:
            gender = gender_encode(row['gender'])
            age = row['age']
            cluster = row['cluster']
            patient_gender[pid] = gender
            patient_age[pid] = int(age)
            patient_cluster[pid] = int(cluster)
    print('\r\t%d in %d rows' % (len(patients), len(patients)))
    return patient_gender, patient_age, patient_cluster


def parse_notes(path, patient_admission: dict) -> dict:
    print('parsing NOTEEVENTS.csv ...')
    notes_path = os.path.join(path, 'NOTEEVENTS.csv')
    notes = pd.read_csv(
        notes_path,
        usecols=['HADM_ID', 'TEXT', 'CATEGORY'],
        converters={'HADM_ID': lambda x: np.int(x) if x != '' else -1, 'TEXT': np.str, 'CATEGORY': np.str}
    )
    patient_note = dict()
    for i, (pid, admissions) in enumerate(patient_admission.items()):
        print('\r\t%d in %d patients' % (i + 1, len(patient_admission)), end='')
        admission_id = admissions[-2]['admission_id']
        note = [row['TEXT'] for _, row in notes[notes['HADM_ID'] == admission_id].iterrows()
                if row['CATEGORY'] == 'Discharge summary']
        note = ' '.join(note)
        admission_id = admissions[-1]['admission_id']
        note2 = [row['TEXT'] for _, row in notes[notes['HADM_ID'] == admission_id].iterrows()
                 if row['CATEGORY'] != 'Discharge summary']
        note2 = ' '.join(note2)
        if len(note) > 0 or len(note2) > 0:
            note = note + ' ' + note2
            patient_note[pid] = note
    print('\r\t%d in %d patients' % (len(patient_admission), len(patient_admission)))
    return patient_note


def calibrate_patient_by_admission(patient_admission: dict, admission_codes: dict):
    print('calibrating patients by admission ...')
    del_pids = []
    for pid, admissions in patient_admission.items():
        for admission in admissions:
            if admission['admission_id'] not in admission_codes:
                break
        else:
            continue
        del_pids.append(pid)
    for pid in del_pids:
        admissions = patient_admission[pid]
        for admission in admissions:
            if admission['admission_id'] in admission_codes:
                del admission_codes[admission['admission_id']]
            else:
                print('\tpatient %d have an admission %d without diagnosis' % (pid, admission['admission_id']))
        del patient_admission[pid]


def calibrate_patient_by_notes(patient_admission: dict, admission_codes: dict, patient_note: dict):
    print('calibrating patients by notes ...')
    del_pids = [pid for pid in patient_admission if pid not in patient_note]
    for pid in del_pids:
        print('\tpatient %d doesn\'t have notes' % pid)
        admissions = patient_admission[pid]
        for admission in admissions:
            del admission_codes[admission['admission_id']]
        del patient_admission[pid]

