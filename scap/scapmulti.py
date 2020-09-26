import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

sys.path.append('..')
from tools.datatools import load_data

# Runs a single n/L combination using multiprocessing
# Warning: This program may crash if there is not enough memory available for each thread.
def generate_profile(code, ngram_size):
    profile = {}
    for i in range(len(code)):
        if i + ngram_size > len(code):
            break
        ngram = code[i:i + ngram_size]
        if ngram not in profile:
            profile[ngram] = 1
        else:
            profile[ngram] += 1
    return profile


def append_profile(current_profile, additional_profile):
    for ngram in additional_profile:
        if ngram not in current_profile:
            current_profile[ngram] = additional_profile[ngram]
        else:
            current_profile[ngram] += additional_profile[ngram]

    return current_profile


def dictionary_to_list(dict):
    return sorted(dict, key=dict.get, reverse=True)


def compare_to_profiles(profile, author_profiles):
    best_match = -1  # num of ngrams in profile and unknown code
    best_profile = -1
    profile = set(profile)
    for author in author_profiles:
        score = len(profile.intersection(author_profiles[author]))
        if score > best_match:
            best_match = score
            best_profile = author
    return best_profile


def main(args):
    start = time.time()
    print('Settings:')
    print(str(args)[10:-1])
    ngram_size = args.ngram

    profile_len = args.features
    author_profiles = {}  # author n-gram profiles 0-999
    # _, train_x, train_y = prep_inputs('train', bytes=True)
    # _, dev_x, dev_y = prep_inputs('dev', bytes=True)
    train_x, train_y, dev_x, dev_y, _, _, _, _ = load_data('../data_dir/', bytes=args.byte,
                                                           preprocess=args.preprocessed)

    print("Dataset loaded.")
    for i in range(len(train_x)):
        single_profile = generate_profile(train_x[i], ngram_size)  # creates profile for each code file

        if train_y[i] in author_profiles:  # appends to existing author profile or creates a new one if it doesn't
            # already exist
            author_profiles[train_y[i]] = append_profile(author_profiles[train_y[i]], single_profile)
        else:
            author_profiles[train_y[i]] = single_profile

    # for author in author_profiles:
    #     author_profiles[author] = set(dictionary_to_list(author_profiles[author])[:profile_len])

    for author in author_profiles:
        if profile_len >= 0:
            author_profiles[author] = set(dictionary_to_list(author_profiles[author])[:profile_len])
        # Each author profile is now a set of the top profile_len number of features.
        elif profile_len == -1:
            auth_dict = author_profiles[author]  # count dictionary for author
            keys = list(auth_dict)  # list of ngrams
            for key in keys:  # if a key only appears once, remove it
                if auth_dict[key] == 1:
                    del auth_dict[key]
            author_profiles[author] = set(dictionary_to_list(author_profiles[author]))
    # lowest=999999999
    # highest=0
    # cx=0
    # for a in author_profiles:
    #     cx+=len(author_profiles[a])
    #     lowest=min(lowest,len(author_profiles[a]))
    #     highest = max(highest, len(author_profiles[a]))
    # print(cx)
    # print(lowest)
    # print(highest)
    # exit()

    print("Author profiles ready.")


    dev_x = [dictionary_to_list(generate_profile(x, ngram_size)) for x in dev_x]
    print('Dev ready for comparisons.')
    start_time = time.time()

    size = int(25000 / cpu_count())

    smaller_chunks = [dev_x[x:x + size] for x in range(0, len(dev_x), size)]
    labels = [dev_y[x:x + size] for x in range(0, len(dev_y), size)]
    with ProcessPoolExecutor() as executor:
        results = [executor.submit(calculate_profiles, smaller_chunks[i], labels[i], author_profiles, i) for i in
                   range(len(smaller_chunks))]

        total = 0
        success = 0
        for r in as_completed(results):
            r = r.result()
            success += r[0]
            total += r[1]
    print(success)
    print(total)
    print(success / total)
    print(time.time() - start_time)


def calculate_profiles(dev_x, dev_y, author_profiles, n):
    print("Starting process {}".format(n))
    count_total = 0
    count_success = 0
    for i in range(len(dev_x)):
        actual = dev_y[i]
        result = compare_to_profiles(dev_x[i], author_profiles)
        count_total += 1
        if actual == result:
            count_success += 1
    print("Finished process {}".format(n))
    return count_success, count_total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('-d',
                        '--data',
                        type=str,
                        default=r'../data_dir',
                        help='Path to the data directory.')

    parser.add_argument('-n',
                        '--ngram',
                        type=int,
                        default=6,
                        help='Profile n-gram size. Default = 6'
                        )
    parser.add_argument('-f',
                        '--features',
                        type=int,
                        default=2000,
                        help='The length of the author profile. Default = 2000'
                        )
    parser.add_argument('-y',
                        '--byte',
                        help='Whether to load the data files as bytes (True) or as strings (False). Default = False',
                        action='store_true')
    parser.add_argument('-p',
                        '--preprocessed',
                        help='Whether to use preprocessed data or not. Default = False.',
                        action='store_true')
    args = parser.parse_args()
    main(args)
