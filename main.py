import datetime
import itertools
import pickle
from functools import reduce

import numpy as np
import collections

from delayed_interrupt import DelayedKeyboardInterrupt
from tools import signum_zero_to_plus, signum_zero_to_minus, get_vectors
from tpm import TPM

import multiprocessing
import logging
import os
from sympy.combinatorics.graycode import GrayCode


def get_similarity(a: TPM, b: TPM):
    return sum(a.weights.flatten() == b.weights.flatten()) / len(a.weights.flatten())


def _uniformize_weights(weights, m):
    weights_freq = {k: 0 for k in range(-m, m + 1)}
    w_changed = []
    for w in weights:
        items = list(weights_freq.items())
        dct_max = max(items, key=lambda x: (x[1], x[0]))
        dct_max_set = {x[0] for x in weights_freq.items() if x[1] == dct_max[1]}
        dct_min = min(items, key=lambda x: (x[1], x[0]))

        if w in dct_max_set:
        # if w == dct_max[0]:
            weights_freq[int(dct_min[0])] += 1
            w_changed.append(dct_min[0])
        else:
            weights_freq[int(w)] += 1
            w_changed.append(w)
    return w_changed


def uniformize_weights(weights, m, iterations=1):
    for _ in range(iterations):
        weights = _uniformize_weights(weights, m)
    return weights


def dropout(weights, entr, bits=4):
    l = 0
    for w in weights:
        if l > 0:
            l -= entr
            continue
        else:
            l = l - entr + bits
            yield w


def optimize(tpm_a: TPM, tpm_b: TPM, weights, max_x, size):
    # vec = np.random.randint(-max_x, max_x + 1, size[0] * size[1])
    lst = list(range(-max_x, max_x + 1))
    lst.remove(0)
    vec = np.random.choice(lst, size[0] * size[1])
    tau_a, _ = tpm_a.get_output(vec)
    tau_b, _ = tpm_b.get_output(vec)

    tpm_a.optimize(vec, tau_b)
    tpm_b.optimize(vec, tau_a)
    return tau_b == tau_a


def evil_optimize(tpm_a: TPM, tpm_b: TPM, evil_tpm: TPM, weights, max_x, size):
    # vec = np.random.randint(-max_x, max_x + 1, size[0] * size[1])
    lst = list(range(-max_x, max_x + 1))
    lst.remove(0)

    vec = np.random.choice(lst, size[0] * size[1])
    tau_a, _ = tpm_a.get_output(vec)
    tau_b, _ = tpm_b.get_output(vec)
    tau_evil, _ = evil_tpm.get_output(vec)

    tpm_a.optimize(vec, tau_b)
    tpm_b.optimize(vec, tau_a)

    if tau_a == tau_b == tau_evil:
        evil_tpm.optimize(vec, tau_a)

    return tau_b == tau_a, tau_a == tau_b == tau_evil


def evil_optimize_many(tpm_a: TPM, tpm_b: TPM, evil_tpms: list[TPM], weights, max_x, size):
    # vec = np.random.randint(-max_x, max_x + 1, size[0] * size[1])
    lst = list(range(-max_x, max_x + 1))
    lst.remove(0)

    vec = np.random.choice(lst, size[0] * size[1])
    tau_a, _ = tpm_a.get_output(vec)
    tau_b, _ = tpm_b.get_output(vec)


    tpm_a.optimize(vec, tau_b)
    tpm_b.optimize(vec, tau_a)
    for evil_tpm in evil_tpms:
        tau_evil, _ = evil_tpm.get_output(vec)
        if tau_a == tau_b == tau_evil:
            evil_tpm.optimize(vec, tau_a)

    return tau_b == tau_a, False


def evil_optimize_advanced(tpm_a: TPM, tpm_b: TPM, evil_tpms: list[TPM], weights, max_x, size):
    # vec = np.random.randint(-max_x, max_x + 1, size[0] * size[1])
    lst = list(range(-max_x, max_x + 1))
    lst.remove(0)

    vec = np.random.choice(lst, size[0] * size[1])
    tau_a, _ = tpm_a.get_output(vec)
    tau_b, _ = tpm_b.get_output(vec)

    tpm_a.optimize(vec, tau_b)
    tpm_b.optimize(vec, tau_a)

    for evil_tpm in evil_tpms:
        tau_evil, _ = evil_tpm.get_output(vec)
        if tau_a == tau_b == tau_evil:
            evil_tpm.optimize(vec, tau_a)
        elif tau_a == tau_b:
            evil_tpm.optimize_adversarial(vec, tau_a)

    return tau_b == tau_a, False


def evil_optimize_cohort(tpm_a: TPM, tpm_b: TPM, evil_tpms, weights, max_x, size):
    # vec = np.random.randint(-max_x, max_x + 1, size[0] * size[1])
    lst = list(range(-max_x, max_x + 1))
    lst.remove(0)

    vec = np.random.choice(lst, size[0] * size[1])
    tau_a, _ = tpm_a.get_output(vec)
    tau_b, _ = tpm_b.get_output(vec)
    # tau_evils = [x.get_output(vec)[0] for x in evil_tpms]

    tpm_a.optimize(vec, tau_b)
    tpm_b.optimize(vec, tau_a)

    if tau_a == tau_b:
        votes = [x.get_vote(vec, tau_a) for x in evil_tpms]
        sigma = collections.Counter([tuple(x) for x in votes]).most_common(1)[0][0]
        [x.optimize_cohort(vec, tau_a, sigma) for x in evil_tpms]

    return tau_b == tau_a, tau_a == tau_b


def evil_optimize_genetic(tpm_a: TPM, tpm_b: TPM, evil_tpms, weights, max_x, size, max_items=50):
    # vec = np.random.randint(-max_x, max_x + 1, size[0] * size[1])
    lst = list(range(-max_x, max_x + 1))
    lst.remove(0)

    vec = np.random.choice(lst, size[0] * size[1])
    tau_a, _ = tpm_a.get_output(vec)
    tau_b, _ = tpm_b.get_output(vec)
    tau_evils = [x.get_output(vec)[0] for x in evil_tpms]
    next_gen = []
    tpm_a.optimize(vec, tau_b)
    tpm_b.optimize(vec, tau_a)

    if tau_a == tau_b:
        for evil_tau, evil_tpm in zip(tau_evils, evil_tpms):
            if evil_tau == tau_a:
                evil_tpm.optimize(vec, tau_a)
                next_gen.append(evil_tpm)
            else:
                for sigma in get_vectors(tau_a):
                    tpm = evil_tpm.clone()
                    tpm.optimize(vec, tau_a, sigma=sigma, this_tau=tau_a)
                    next_gen.append(tpm)
    else:
        next_gen = evil_tpms

    if len(next_gen) > max_items:
        next_gen = sorted(next_gen, key=lambda x: get_similarity(x, tpm_a), reverse=True)

    return tau_b == tau_a, tau_a == tau_b, next_gen[:max_items]


def perform_key_agreement(k, n, l, max_x):
    size = (k, n)

    tpm_a = TPM(*(k, n, l), signum=signum_zero_to_plus)
    tpm_b = TPM(*(k, n, l), signum=signum_zero_to_minus)

    _iterations = 0
    _optimizations = 0
    while not np.array_equal(tpm_b.weights, tpm_a.weights):
        _iterations += 1
        _optimizations += int(optimize(tpm_a, tpm_b, l, max_x, size))

    return _iterations, _optimizations, tpm_a.weights.flatten()


def perform_adversarial_key_agreement_cohort(k, n, l, max_x):
    size = (k, n)
    similarity = []
    similarity_evil = []
    tpm_a = TPM(*(k, n, l), signum=signum_zero_to_plus)
    tpm_b = TPM(*(k, n, l), signum=signum_zero_to_minus)

    length = len(tpm_a.weights.flatten())

    tpms_evil = [TPM(*(k, n, l), signum=signum_zero_to_minus) for _ in range(50)]

    _iterations = 0

    _evil_optimizations = 0
    _optimizations = 0
    while not np.array_equal(tpm_b.weights, tpm_a.weights):
        _iterations += 1
        _normal_optimization, _evil_optimization = evil_optimize_cohort(tpm_a, tpm_b, tpms_evil, l, max_x, size)
        normal_opt = int(_normal_optimization)
        _optimizations += normal_opt
        _evil_optimizations += int(_evil_optimization)
        if normal_opt:
            similarity.append(sum(tpm_b.weights.flatten() == tpm_a.weights.flatten()) / length)
            similarity_evil.append(max(
                max(get_similarity(tpm_a, tpm_evil), get_similarity(tpm_b, tpm_evil)) for tpm_evil in tpms_evil
            ))
    return _iterations, _optimizations, _evil_optimizations, tpm_a.weights, similarity, \
        similarity_evil, uniformize_weights(tpm_a.weights, l)


def perform_adversarial_key_agreement_genetic(k, n, l, max_x):
    size = (k, n)
    similarity = []
    similarity_evil = []
    tpm_a = TPM(*(k, n, l), signum=signum_zero_to_plus)
    tpm_b = TPM(*(k, n, l), signum=signum_zero_to_minus)

    length = len(tpm_a.weights.flatten())

    tpms_evil = [TPM(*(k, n, l), signum=signum_zero_to_minus) for _ in range(5)]

    _iterations = 0

    _evil_optimizations = 0
    _optimizations = 0
    while not np.array_equal(tpm_b.weights, tpm_a.weights):
        _iterations += 1
        _normal_optimization, _evil_optimization, tpms_evil = evil_optimize_genetic(tpm_a, tpm_b, tpms_evil, l, max_x,
                                                                                    size)
        normal_opt = int(_normal_optimization)
        _optimizations += normal_opt
        _evil_optimizations += int(_evil_optimization)
        if normal_opt:
            similarity.append(sum(tpm_b.weights.flatten() == tpm_a.weights.flatten()) / length)
            similarity_evil.append(max(
                max(get_similarity(tpm_a, tpm_evil), get_similarity(tpm_b, tpm_evil)) for tpm_evil in tpms_evil
            ))
    return _iterations, _optimizations, _evil_optimizations, tpm_a.weights, similarity, \
        similarity_evil, uniformize_weights(tpm_a.weights, l)


def perform_adversarial_key_agreement(k, n, l, max_x):
    size = (k, n)
    similarity = []
    similarity_evil = []
    tpm_a = TPM(*(k, n, l), signum=signum_zero_to_plus)
    tpm_b = TPM(*(k, n, l), signum=signum_zero_to_minus)

    length = len(tpm_a.weights.flatten())

    tpm_evil = TPM(*(k, n, l), signum=signum_zero_to_minus)

    _iterations = 0

    _evil_optimizations = 0
    _optimizations = 0
    while not np.array_equal(tpm_b.weights, tpm_a.weights):
        _iterations += 1
        _normal_optimization, _evil_optimization = evil_optimize(tpm_a, tpm_b, tpm_evil, l, max_x, size)
        normal_opt = int(_normal_optimization)
        _optimizations += normal_opt
        _evil_optimizations += int(_evil_optimization)
        if normal_opt:
            similarity.append(sum(tpm_b.weights.flatten() == tpm_a.weights.flatten()) / length)
            similarity_evil.append(max(
                sum(tpm_evil.weights.flatten() == tpm_a.weights.flatten()) / length,
                sum(tpm_evil.weights.flatten() == tpm_a.weights.flatten()) / length
            ))

    return _iterations, _optimizations, _evil_optimizations, tpm_a.weights, tpm_evil.weights, similarity, similarity_evil, uniformize_weights(
        tpm_a.weights, l)


def perform_adversarial_key_agreement_many(k, n, l, max_x, evil_tpms=1):
    size = (k, n)
    similarity = []
    similarity_evil = []
    tpm_a = TPM(*(k, n, l), signum=signum_zero_to_plus)
    tpm_b = TPM(*(k, n, l), signum=signum_zero_to_minus)

    length = len(tpm_a.weights.flatten())

    tpms_evil = [TPM(*(k, n, l), signum=signum_zero_to_minus) for _ in range(evil_tpms)]

    _iterations = 0

    _evil_optimizations = 0
    _optimizations = 0
    while not np.array_equal(tpm_b.weights, tpm_a.weights):
        _iterations += 1
        _normal_optimization, _evil_optimization = evil_optimize_many(tpm_a, tpm_b, tpms_evil, l, max_x, size)
        normal_opt = int(_normal_optimization)
        _optimizations += normal_opt
        _evil_optimizations += int(_evil_optimization)
        if normal_opt:
            similarity.append(sum(tpm_b.weights.flatten() == tpm_a.weights.flatten()) / length)
            similarity_evil.append(max(
                max(get_similarity(tpm_a, tpm_evil), get_similarity(tpm_b, tpm_evil)) for tpm_evil in tpms_evil
            ))


    return _iterations, _optimizations, _evil_optimizations, tpm_a.weights, None, similarity, similarity_evil, None

def perform_adversarial_key_agreement_advanced(k, n, l, max_x, evil_tpms=50):
    size = (k, n)
    similarity = []
    similarity_evil = []
    tpm_a = TPM(*(k, n, l), signum=signum_zero_to_plus)
    tpm_b = TPM(*(k, n, l), signum=signum_zero_to_minus)

    length = len(tpm_a.weights.flatten())

    tpms_evil = [TPM(*(k, n, l), signum=signum_zero_to_minus) for _ in range(evil_tpms)]

    _iterations = 0

    _evil_optimizations = 0
    _optimizations = 0
    while not np.array_equal(tpm_b.weights, tpm_a.weights):
        _iterations += 1
        _normal_optimization, _evil_optimization = evil_optimize_advanced(tpm_a, tpm_b, tpms_evil, l, max_x, size)
        normal_opt = int(_normal_optimization)
        _optimizations += normal_opt
        _evil_optimizations += int(_evil_optimization)
        if normal_opt:
            similarity.append(sum(tpm_b.weights.flatten() == tpm_a.weights.flatten()) / length)
            similarity_evil.append(max(
                max(get_similarity(tpm_a, tpm_evil), get_similarity(tpm_b, tpm_evil)) for tpm_evil in tpms_evil
            ))

    return _iterations, _optimizations, _evil_optimizations, tpm_a.weights, None, similarity, similarity_evil, uniformize_weights(
        tpm_a.weights, l)


def setup_logger():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)


def save_result_func(lst, lock: multiprocessing.Lock()):
    def func(result):
        lock.acquire()
        try:
            lst.append(result)
            if len(lst) % 10 == 0:
                logging.info("saving {} result".format(len(lst)))
        finally:
            lock.release()

    return func


class Scenario:
    def __init__(self, func):
        self.k = 3

        self.iterations = 100
        self.func = func
        self.params_n = [50,60,70]
        self.params_max_x = [1,2,3,4,5]
        # self.params_max_x = [1]
        self.params_max_weight = [8]
        self.datetime = int(datetime.datetime.now().timestamp())

    def run(self):
        os.mkdir(f"results/{self.datetime}")
        for (n, max_x, max_weight) in itertools.product(self.params_n, self.params_max_x, self.params_max_weight):

            logging.info("Starting simulation with N:{} and MAX_X:{}, MAX_WEIGHT:{} and FUNC:{}"
                         .format(n, max_x, max_weight, self.func.__name__))

            results = []
            weights_dict = {k: 0 for k in range(-max_weight + 1, max_weight + 1)}

            # logging.info("Starting simulation")
            lock = multiprocessing.Lock()
            save_result = save_result_func(results, lock)
            try:
                with multiprocessing.Pool() as pool:
                    for no in range(self.iterations):
                        pool.apply_async(self.func, args=(self.k, n, max_weight, max_x), callback=save_result)
                        # ret = perform_key_agreement(k, n, max_weight)
                        # iteration, optimization, weights = ret

                        # with DelayedKeyboardInterrupt():
                        #     iterations.append(iteration)
                        #     optimizations.append(optimization)
                        #     for weight_line in weights:
                        #         for weight in weight_line:
                        #             weights_dict[weight] += 1
                    pool.close()
                    pool.join()

                    #
                    # if no == max_iterations:
                    #     logging.info("Finishing simulation")
                    #     break

            finally:

                iterations_lst = []
                optimizations_lst = []
                evil_optimizations_lst = []
                similarity_lst = []
                weights_dict = {k: 0 for k in range(-max_weight, max_weight + 1)}
                changed_weights_dict = {k: 0 for k in range(-max_weight, max_weight + 1)}
                similarities = []
                evil_similarities = []
                # for iterations, optimizations, evil_optimizations, weights, evil_weights, similarity, evil_similarity, changed_weights in results:
                #     iterations_lst.append(iterations)
                #     optimizations_lst.append(optimizations)
                #     evil_optimizations_lst.append(evil_optimizations)
                #     similarity_lst.append(int(sum(evil_weights.flatten() == weights.flatten())))
                #     similarities.append(similarity)
                #     evil_similarities.append(evil_similarity)
                #     for val, occurrences in collections.Counter(weights.flatten()).items():
                #         weights_dict[val] += occurrences
                #
                #     for val, occurrences in collections.Counter(changed_weights).items():
                #         changed_weights_dict[val] += occurrences

                with open('results/{}/_results-X{}-N{}-MAX_WEIGHT{}-func-{}.pickle'.format(self.datetime, max_x, n,
                                                                                           max_weight,
                                                                                           self.func.__name__),
                          'wb') as f:
                    # pickle.dump((
                    #             iterations_lst, optimizations_lst, weights_dict, evil_optimizations_lst, similarity_lst,
                    #             similarities, evil_similarities, changed_weights_dict), f)
                    pickle.dump(results, f)

                logging.info("Saving results")
                # logging.info(f"Avg: {sum(optimizations_lst) / len(optimizations_lst)}")


if __name__ == "__main__":
    setup_logger()
    # scenario = Scenario(perform_adversarial_key_agreement_advanced)
    # scenario.run()
    # scenario = Scenario(perform_adversarial_key_agreement_many)
    # scenario.run()
    scenario = Scenario(perform_key_agreement)
    scenario.run()
    # scenario = Scenario(perform_adversarial_key_agreement_cohort)
    # scenario.run()
    # scenario = Scenario(perform_adversarial_key_agreement_genetic)
    # scenario.run()
