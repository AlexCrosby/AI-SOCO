import argparse
import sys
import time

sys.path.append('..')
from tools.datatools import load_data


# Runs a single n/L combination.
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
        score = len(profile.intersection(
            author_profiles[author]))  # Gets the intersection score from each profile and saves best.
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
    for author in author_profiles:
        if profile_len >= 0:
            author_profiles[author] = set(dictionary_to_list(author_profiles[author])[:profile_len])
        # Each author profile is now a set of the top profile_len number of features.
        elif profile_len == -1:
            print("HYPER")
            auth_dict = author_profiles[author]  # count dictionary for author
            keys = list(auth_dict)  # list of ngrams
            for key in keys:  # if a key only appears once, remove it
                if auth_dict[key] == 1:
                    del auth_dict[key]
            author_profiles[author] = set(dictionary_to_list(author_profiles[author]))

    print("Author profiles ready.")

    count_total = 0
    count_success = 0

    dev_x = [dictionary_to_list(generate_profile(x, ngram_size)) for x in dev_x]
    print('Dev ready for comparisons.')
    start_time = time.time()
    for i in range(len(dev_x)):

        actual = dev_y[i]
        result = compare_to_profiles(dev_x[i], author_profiles)
        count_total += 1
        if actual == result:
            count_success += 1
        if ((i + 1) % 250) == 0:
            percent = int((i + 1) / 250)
            print("Progress: {}%".format(percent))
            print("Accuracy so far: {}".format(count_success / count_total))
            time_secs = ((time.time() - start_time) / percent) * (100 - percent)
            time_mins = int(time_secs // 60)
            time_secs = str(int(time_secs % 60)).zfill(2)
            print("Time remaining: {}:{}".format(time_mins, time_secs))
    print('Total Guesses: {}'.format(count_total))
    print('Correct Guesses: {}'.format(count_success))
    print('Guess accuracy: {}'.format(count_success / count_total))
    print('n-grams: {}'.format(ngram_size))
    print('Profile length: {}'.format(profile_len))
    print(time.time() - start)


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
