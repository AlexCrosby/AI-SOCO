import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

sys.path.append('..')
from tools.datatools import load_data

# Runs a multiple n/L combinations using multiprocessing
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
    print('Settings:')
    print(str(args)[10:-1])
    train_x, train_y, dev_x, dev_y, _, _, _, _ = load_data(args.data, bytes=args.byte, preprocess=args.preprocessed)
    dev_orig = dev_x.copy()  # [:2500]  # To do shorter set, uncomment [:2500]
    print("Dataset loaded.")
    ngram_sizes = [int(n) for n in args.ngrams.split()]
    profile_lens = [int(n) for n in args.features.split()]
    print(ngram_sizes)
    print(profile_lens)

    for ngram_size in ngram_sizes:
        author_profiles = {}
        dev_x = dev_orig.copy()
        for i in range(len(train_x)):
            single_profile = generate_profile(train_x[i], ngram_size)

            if train_y[i] in author_profiles:
                author_profiles[train_y[i]] = append_profile(author_profiles[train_y[i]], single_profile)
            else:
                author_profiles[train_y[i]] = single_profile
        author_profiles_backup = author_profiles.copy()

        dev_x = [dictionary_to_list(generate_profile(x, ngram_size)) for x in dev_x]

        #############################

        for profile_len in profile_lens:
            author_profiles = author_profiles_backup.copy()  # First load all the author ngrams and fix to the correct length
            for author in author_profiles:
                if profile_len >= 0:
                    author_profiles[author] = set(dictionary_to_list(author_profiles[author])[:profile_len])
                # Each author profile is now a set of the top profile_len number of features.
                elif profile_len == -1:
                    # print("HYPER")
                    auth_dict = author_profiles[author]  # count dictionary for author
                    keys = list(auth_dict)  # list of ngrams
                    for key in keys:  # if a key only appears once, remove it
                        if auth_dict[key] == 1:
                            del auth_dict[key]
                    author_profiles[author] = set(dictionary_to_list(author_profiles[author]))
            print("Running {}@{}".format(ngram_size, profile_len))

            start_time = time.time()
            processes = cpu_count()
            size = int(25000 / processes)

            smaller_chunks = [dev_x[x:x + size] for x in range(0, len(dev_x), size)]
            labels = [dev_y[x:x + size] for x in range(0, len(dev_y), size)]
            with ProcessPoolExecutor() as executor:
                results = [executor.submit(calculate_profiles, smaller_chunks[i], labels[i], author_profiles) for i in
                           range(len(smaller_chunks))]

                total = 0
                success = 0
                for r in as_completed(results):
                    r = r.result()
                    success += r[0]
                    total += r[1]
                executor.shutdown(wait=True)

            # print(success)
            # print(total)
            # print(success / total)
            print(time.time() - start_time)
            print('Total Guesses: {}'.format(total))
            print('Correct Guesses: {}'.format(success))
            print('Guess accuracy: {}'.format(success / total))
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


def calculate_profiles(dev_x, dev_y, author_profiles):
    count_total = 0
    count_success = 0
    for i in range(len(dev_x)):
        actual = dev_y[i]
        result = compare_to_profiles(dev_x[i], author_profiles)
        count_total += 1
        if actual == result:
            count_success += 1
    return count_success, count_total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('-d',
                        '--data',
                        type=str,
                        default=r'../data_dir',
                        help='Path to the data directory.')

    parser.add_argument('ngrams',
                        type=str,
                        help='List of ngrams to calculate.'
                        )
    parser.add_argument('features',
                        type=str,
                        help='List of profile lengths to calculate for each ngram.'
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
