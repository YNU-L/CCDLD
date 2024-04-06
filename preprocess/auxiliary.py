import os
import numpy as np
import math


def parse_icd9_range(range_: str) -> (str, str, int, int):
    ranges = range_.lstrip().split('-')
    if ranges[0][0] == 'V':
        prefix = 'V'
        format_ = '%02d'
        start, end = int(ranges[0][1:]), int(ranges[1][1:])
    elif ranges[0][0] == 'E':
        prefix = 'E'
        format_ = '%03d'
        start, end = int(ranges[0][1:]), int(ranges[1][1:])
    else:
        prefix = ''
        format_ = '%03d'
        if len(ranges) == 1:
            start = int(ranges[0])
            end = start + 1
        else:
            start, end = int(ranges[0]), int(ranges[1])
    return prefix, format_, start, end


def generate_code_levels(path, code_map: dict) -> np.ndarray:
    print('generating code levels ...')
    three_level_code_set = set(code.split('.')[0] for code in code_map)
    icd9_path = os.path.join(path, 'icd9.txt')
    icd9_range = list(open(icd9_path, 'r', encoding='utf-8').readlines())
    three_level_dict = dict()
    level1, level2, level3 = (1, 1, 1)
    level1_can_add = False
    for range_ in icd9_range:
        range_ = range_.rstrip()
        if range_[0] == ' ':
            prefix, format_, start, end = parse_icd9_range(range_)
            level2_cannot_add = True
            for i in range(start, end + 1):
                code = prefix + format_ % i
                if code in three_level_code_set:
                    three_level_dict[code] = [level1, level2, level3]
                    level3 += 1
                    level1_can_add = True
                    level2_cannot_add = False
            if not level2_cannot_add:
                level2 += 1
        else:
            if level1_can_add:
                level1 += 1
                level1_can_add = False

    level4 = 1
    code_level = dict()
    for code in code_map:
        three_level_code = code.split('.')[0]
        if three_level_code in three_level_dict:
            three_level = three_level_dict[three_level_code]
            code_level[code] = three_level + [level4]
            level4 += 1
        else:
            print(three_level_code)
            code_level[code] = [0, 0, 0, 0]

    code_level_matrix = np.zeros((len(code_map) + 1, 4), dtype=int)
    for code, cid in code_map.items():
        code_level_matrix[cid] = code_level[code]

    return code_level_matrix


def generate_patient_code_adjacent(code_x: np.ndarray, code_num: int) -> np.ndarray:
    print('generating patient code adjacent matrix ...')
    result = np.zeros((len(code_x), code_num + 1), dtype=int)
    for i, codes in enumerate(code_x):
        adj_codes = codes[codes > 0]
        result[i][adj_codes] = 1
    return result


def generate_code_code_adjacent(code_num: int, code_level_matrix: np.ndarray) -> np.ndarray:
    print('generating code code adjacent matrix ...')
    n = code_num + 1
    result = np.zeros((n, n), dtype=int)
    for i in range(1, n):
        print('\r\t%d / %d' % (i, n), end='')
        for j in range(1, n):
            if i != j:
                level_i = code_level_matrix[i]
                level_j = code_level_matrix[j]
                same_level = 4
                while same_level > 0:
                    level = same_level - 1
                    if level_i[level] == level_j[level]:
                        break
                    same_level -= 1
                result[i, j] = same_level + 1
    print('\r\t%d / %d' % (n, n))
    return result


def co_occur(pids: np.ndarray,
              patient_admission: dict,
              admission_codes_encoded: dict,
              code_num: int) -> np.ndarray:
    print('calculating co-occurrence ...')
    x = np.zeros((code_num + 1, code_num + 1), dtype=float)
    co_occur_counts = np.zeros((code_num + 1, code_num + 1), dtype=float)
    for i, pid in enumerate(pids):
        print('\r\t%d / %d' % (i + 1, len(pids)), end='')
        admissions = patient_admission[pid]
        for k, admission in enumerate(admissions[:-1]):
            codes = admission_codes_encoded[admission['admission_id']]
            for m in range(len(codes) - 1):
                for n in range(m + 1, len(codes)):
                    c_i, c_j = codes[m], codes[n]
                    co_occur_counts[c_i, c_j] += 1
                    co_occur_counts[c_j, c_i] += 1
    print("\r\t%d / %d" % (len(pids), len(pids)))

    y = count_connection(co_occur_counts)
    y = remove_zero(y)
    threshold = cal_threshold(y)
    for i in range(co_occur_counts.shape[0]):
        print('\r\t%d / %d' % (i + 1, code_num + 1), end='')
        for j in range(co_occur_counts.shape[1]):
            if co_occur_counts[i, j] >= threshold:
                x[i, j] = 1
                x[j, i] = 1
    print('\r\t%d / %d\n' % (code_num + 1, code_num + 1), end='')
    return x


def count_connection(x: np.ndarray) -> dict:
    unique, counts = np.unique(x, return_counts=True)
    return dict(zip(unique, counts))


def remove_zero(counts_dict):
    if 0 in counts_dict:
        del counts_dict[0]
    return counts_dict


def cal_threshold(data):
    numbers = list(data.keys())
    frequencies = list(data.values())
    N = sum(frequencies)

    miu = sum(num * freq for num, freq in zip(numbers, frequencies)) / N
    sigma_square = sum(freq * ((num - miu) ** 2) for num, freq in zip(numbers, frequencies)) / N
    sigma = math.sqrt(sigma_square)

    ita_sum = 0
    for i in range(math.floor(miu)):
        ita_sum = ita_sum + data.get(i + 1)
    ita = (N - ita_sum) / N
    delta = math.ceil(miu + ita * sigma)

    return delta

