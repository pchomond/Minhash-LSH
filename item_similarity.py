import csv
from collections import namedtuple
from typing import Dict, List
import random
from operator import itemgetter
import sys
import heapq
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

# ------------------------------------------- #
# ------------------------------------------- #
# -------------Global Variables-------------- #
# ------------------------------------------- #
# ------------------------------------------- #

user_list: Dict[str, List[str]] = {}
movie_map: Dict[str, int] = {}
movie_list: Dict[str, List[str]] = {}

user_counter = 1
movie_counter = 1

false_positives_matrix: List[int] = []
false_negatives_matrix: List[int] = []
precision_matrix: List[float] = []
recall_matrix: List[float] = []
f1_scores_matrix: List[float] = []

# -------------------- #
# Experiment Variables #
# -------------------- #
DATASET_FILENAME = 'ratings_100users.csv'
NO_OF_HASH_FUNCTIONS = 40
NO_OF_CONSIDERED_MOVIES = 100
ACCEPTANCE_LEVEL = 0.25

# Min Hashing #
n_values = [5, 10, 15, 20, 25, 30, 35, 40]
# n_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

# Locality Sensitive Hashing (Must n = b * r) #
n = 40
b_values = [20, 10, 8, 5, 4, 2]
r_values = [2, 4, 5, 8, 10, 20]


# ------------------------------------------- #
# ------------------------------------------- #
# ----------------Functions------------------ #
# ------------------------------------------- #
# ------------------------------------------- #


def get_csv_data(filename):
    with open(filename, 'r', encoding='utf-8-sig') as file_handler:
        reader = csv.reader(file_handler)
        Entry = namedtuple('Entry', next(reader))
        for line in map(Entry._make, reader):
            yield line


def add_to_user_list(row):
    global user_counter, user_list
    if row.userId not in user_list:
        user_list[row.userId] = [row.movieId]
        user_counter += 1
    else:
        user_list[row.userId].append(row.movieId)
    return


def add_to_movie_map(row):
    global movie_counter, movie_map
    if row.movieId not in movie_map:
        movie_map[row.movieId] = movie_counter
        movie_counter += 1
    return


def add_to_movie_list(row):
    global movie_list
    if row.movieId not in movie_list:
        movie_list[row.movieId] = [row.userId]
    else:
        movie_list[row.movieId].append(row.userId)
    return


def init_preprocessing(file_iter):
    for row in file_iter:
        add_to_user_list(row)
        add_to_movie_map(row)
        add_to_movie_list(row)
    return


def jaccard_similarity(movie_id1, movie_id2):
    s1 = set(movie_list[str(movie_id1)])
    s2 = set(movie_list[str(movie_id2)])
    intersection = s1 & s2
    union = s1 | s2
    jaccard_sim = len(intersection) / len(union)
    return jaccard_sim


def create_random_hash_function(p=2 ** 33 - 355, m=2 ** 32 - 1):
    a = random.randint(1, p - 1)
    b = random.randint(0, p - 1)
    return lambda x: 1 + (((a * x + b) % p) % m)


def create_random_permutation(k):
    hash_function = create_random_hash_function(m=k)

    hash_list = []

    for i in range(1, k + 1):
        j = int(hash_function(i))
        hash_list.append((i, j))

    # sort the hashList by second argument of the pairs...
    sorted_hash_list = sorted(hash_list, key=itemgetter(1))

    random_permutation = [i[0] for i in sorted_hash_list]

    return random_permutation


def get_min_hashed_index(random_permutation, user_set):
    return_val = 0
    for x in random_permutation:
        if x in user_set:  # O(1) Lookup time
            return_val = x
            break
    return return_val


def min_hash(n_functions):
    signature_matrix = [[0 for _ in range(movie_counter)] for _ in range(n_functions)]
    for i in range(0, n):  # <- O(n): For each hashing function
        random_permutation = create_random_permutation(user_counter)
        for movie_id, users in movie_list.items():  # <- O(N): Scan M matrix column by column
            # While converting a list to a set creates an overhead,
            # the lookup time for operation 'in' in a set compensates nicely.
            users = list(map(int, users))
            user_set = set(users)
            min_value = get_min_hashed_index(random_permutation, user_set)  # <- Worst case: O(K)
            signature_matrix[i][movie_map[movie_id] - 1] = min_value
    return signature_matrix


def signature_similarity(movie_id1, movie_id2, signature_matrix, no_of_signatures_considered):
    mapped_id1 = movie_map[str(movie_id1)] - 1  # Sanitised movie_id1
    mapped_id2 = movie_map[str(movie_id2)] - 1  # Sanitised movie_id2
    col1 = [row[mapped_id1] for i, row in enumerate(signature_matrix) if i < no_of_signatures_considered]
    col2 = [row[mapped_id2] for i, row in enumerate(signature_matrix) if i < no_of_signatures_considered]
    same_id = sum([1 for x, y in zip(col1, col2) if x == y])
    return same_id / no_of_signatures_considered


def add_leading_zero_to_single_digits(num):
    a = str(num)  # Sanitise num
    if len(a) == 1:
        a = '0' + a
    return a


def convert_sig_to_num(band_signatures):
    band_signatures = map(add_leading_zero_to_single_digits, band_signatures)
    band_string = ''.join(map(str, band_signatures))
    return int(band_string)


# Mutable object 'pairs' will be updated, as it is passed by reference
# We avoid a huge load of set unions creating an even larger overhead
def _generate_pairs(bucket, new_id, pairs):
    for old_id in bucket:
        tup = (old_id, new_id)
        pairs.add(tuple(sorted(tup)))
    return


def lsh(b, r, signature_matrix):
    if n / b != r:
        print("The number of bands does not divide the signature vector evenly. Exiting...", file=sys.stderr)
        sys.exit(1)

    hash_function = create_random_hash_function()
    pairs = set()
    for band in range(0, b):
        buckets = {}
        for col in movie_map:
            band_signatures = []
            for row in range(band * r, r * (band + 1)):
                band_signatures.append(signature_matrix[row][movie_map[col] - 1])
            band_num = convert_sig_to_num(band_signatures)
            hash_value = hash_function(band_num)
            if hash_value not in buckets:
                buckets[hash_value] = [col]
            else:
                _generate_pairs(buckets[hash_value], col, pairs)
                buckets[hash_value].append(col)
    return pairs


def get_k_smallest_keys(k):
    k_keys_sorted = heapq.nsmallest(k, list(map(int, movie_map.keys())))
    k_keys_sorted = list(map(str, k_keys_sorted))
    return k_keys_sorted


def calculate_baseline_j_sim(dataset_indices):
    baseline_results = {}
    ground_truth_count = 0
    for i in range(0, len(dataset_indices)):
        for j in range(i + 1, len(dataset_indices)):
            j_sim = jaccard_similarity(dataset_indices[i], dataset_indices[j])
            baseline_results[(dataset_indices[i], dataset_indices[j])] = j_sim
            if j_sim >= ACCEPTANCE_LEVEL:
                ground_truth_count += 1
    return baseline_results, ground_truth_count


def calculate_pairwise_sig_sim(dataset_indices, baseline_j_sim, signature_matrix, n_value):
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for i in range(0, len(dataset_indices)):
        for j in range(i + 1, len(dataset_indices)):
            sig_sim = signature_similarity(dataset_indices[i], dataset_indices[j], signature_matrix, n_value)
            if baseline_j_sim[(dataset_indices[i], dataset_indices[j])] >= 0.5:
                if sig_sim >= ACCEPTANCE_LEVEL:
                    true_positives += 1
                else:
                    false_negatives += 1
            else:
                if sig_sim >= ACCEPTANCE_LEVEL:
                    false_positives += 1
                else:
                    true_negatives += 1
    return true_positives, true_negatives, false_positives, false_negatives


def calculate_statistics(true_positives, false_positives, false_negatives):
    if true_positives == 0 and false_positives == 0:
        precision = 0
    else:
        precision = true_positives / (true_positives + false_positives)
    if true_positives == 0 and false_negatives == 0:
        recall = 0
    else:
        recall = true_positives / (true_positives + false_negatives)
    if precision == 0 and recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * recall * precision / (recall + precision)

    return precision, recall, f1_score


def handle_statistics(true_positives, true_negatives, false_positives, false_negatives):
    global false_positives_matrix, false_negatives_matrix, precision_matrix, recall_matrix, f1_scores_matrix
    print("True positives: %d" % true_positives)
    print("False negatives: %d" % false_negatives)
    print("False positives: %d" % false_positives)
    print("True negatives: %d" % true_negatives)

    precision, recall, f1_score = calculate_statistics(true_positives, false_positives, false_negatives)

    false_positives_matrix.append(false_positives)
    false_negatives_matrix.append(false_negatives)
    precision_matrix.append(precision)
    recall_matrix.append(recall)
    f1_scores_matrix.append(f1_score)

    print("Precision: %f" % precision)
    print("Recall: %f" % recall)
    print("F1 Measure: %f" % f1_score)
    return


def multi_plot(x_label, x_tick_labels, plot_title):
    gs = gridspec.GridSpec(2, 2)
    x_ticks = [i for i in range(0, len(x_tick_labels))]

    plt.figure()
    _ = plt.subplot(gs[0, 0])
    plt.title(plot_title)
    plt.plot(false_positives_matrix, color='blue')
    plt.xlabel(x_label)
    plt.ylabel('False Positive')
    plt.xticks(x_ticks, x_tick_labels, fontsize='8')

    _ = plt.subplot(gs[0, 1])
    plt.title(plot_title)
    plt.plot(false_negatives_matrix, color='blue')
    plt.xlabel(x_label)
    plt.ylabel('False Negative')
    plt.xticks(x_ticks, x_tick_labels, fontsize='8')

    ax = plt.subplot(gs[1, :])
    ax.set_title(plot_title)
    plt.xlabel(x_label)
    plt.ylabel('Percentage')
    ax.plot(precision_matrix, label='Precision')
    ax.plot(recall_matrix, label='Recall')
    ax.plot(f1_scores_matrix, label='F1-Score')
    ax.legend()
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels, fontsize='8')
    plt.show()
    return


def _reset_matrices():
    global false_positives_matrix, false_negatives_matrix, precision_matrix, recall_matrix, f1_scores_matrix
    false_positives_matrix = []
    false_negatives_matrix = []
    precision_matrix = []
    recall_matrix = []
    f1_scores_matrix = []
    return


def min_hash_experimentation(signature_matrix):
    k_smallest_movie_ids = get_k_smallest_keys(NO_OF_CONSIDERED_MOVIES)
    baseline_results, ground_truth_count = calculate_baseline_j_sim(k_smallest_movie_ids)
    _reset_matrices()

    for n_value in n_values:
        print("\n" + "=" * 25)
        print("Testing for n: %d" % n_value)
        print("=" * 25)

        true_positives, true_negatives, false_positives, false_negatives = \
            calculate_pairwise_sig_sim(k_smallest_movie_ids, baseline_results, signature_matrix, n_value)

        handle_statistics(true_positives, true_negatives, false_positives, false_negatives)

    multi_plot('Number of Signatures', n_values, 'Min Hashing')
    return


def lsh_experimentation(signature_matrix):
    k_smallest_movie_ids = get_k_smallest_keys(NO_OF_CONSIDERED_MOVIES)
    baseline_results, ground_truth_count = calculate_baseline_j_sim(k_smallest_movie_ids)
    _reset_matrices()

    for b, r in zip(b_values, r_values):
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0

        print("\n" + "=" * 25)
        print("Testing for b: %d and r: %d" % (b, r))
        print("=" * 25)

        candidate_pairs = lsh(b, r, signature_matrix)
        for pair, j_sim in baseline_results.items():
            if j_sim >= ACCEPTANCE_LEVEL:
                if pair in candidate_pairs:
                    true_positives += 1
                else:
                    false_negatives += 1
            else:
                if pair in candidate_pairs:
                    false_positives += 1
                else:
                    true_negatives += 1

        handle_statistics(true_positives, true_negatives, false_positives, false_negatives)

    multi_plot('(Bands, Lines)', list(map(str, zip(b_values, r_values))), 'Locality Sensitive Hashing')
    return


def write_to_csv(filename, data):
    with open(filename, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerows(data)
    return


def write_dict_to_csv(filename, data):
    with open(filename, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for key, value in data.items():
            csv_writer.writerow([key, value])
    return


def experimentation():
    sig = min_hash(NO_OF_HASH_FUNCTIONS)
    write_to_csv('signature_matrix.csv', sig)
    start = time.perf_counter()
    min_hash_experimentation(sig)
    middle = time.perf_counter()
    print('\n' + '='*50)
    print("MinHash Experimentation took: %0.3f seconds" % (middle - start))
    print('=' * 50)
    lsh_experimentation(sig)
    end = time.perf_counter()
    print('\n' + '=' * 50)
    print("LSH Experimentation took: %0.3f seconds" % (end - middle))
    print('=' * 50)
    return


def main():
    if len(sys.argv) == 2:
        file_iter = iter(get_csv_data(sys.argv[1]))
    else:
        file_iter = iter(get_csv_data(DATASET_FILENAME))
    init_preprocessing(file_iter)
    write_dict_to_csv('user_list.csv', user_list)
    write_dict_to_csv('movie_map.csv', movie_map)
    write_dict_to_csv('movie_list.csv', movie_list)
    experimentation()
    return


if __name__ == '__main__':
    main()
