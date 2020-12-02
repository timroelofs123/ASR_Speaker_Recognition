import glob
import os
import math


def main():
    """ 
    File used for generating the training and testing files.

    1. Obtain the names of the all the recordings.
    2. Obtain the id of the speaker for each recording.
    3. Make a dictionary which counts how many recordings each speaker has.
    4. Based on the number of recordings per speaker we take 20% into test set and the rest of 80% in training set.

    The result of running this file is two .txt files used for training and testing the models, while having a consistent test set.
    """

    names = []
    for filename in glob.iglob('./LibriSpeech/dev-clean/**', recursive=True):
        if os.path.isfile(
                filename) and '.flac' in filename:  # filter dirs an txt files
            name = '.' + filename.split('.')[1]
            names.append(name)

    names_splitted = []
    for name in names:
        new_name = name.split('/')[5]  # Obtain the id of the speaker
        names_splitted.append(new_name)

    name_dict = {}
    for name in names_splitted:
        speaker = name.split('-')[0]
        if speaker not in name_dict.keys():
            name_dict.update({
                speaker: 1
            })  # If speaker not in dictionary, insert it with 1 recording
        else:
            name_dict[
                speaker] += 1  # Else add another recording to the speaker

    with open('train.txt', 'w') as train_file:
        with open('test.txt', 'w') as test_file:
            # Dictionary used for counting how many recordings we introduced in test set for each speaker
            added_dict = {}
            for name in names_splitted:
                speaker = name.split('-')[0]
                threshold = math.ceil(
                    name_dict[speaker] *
                    0.2)  # Obtain the 20% threshold for each speaker
                if speaker not in added_dict.keys():
                    added_dict.update({speaker: 1})
                    test_file.write('{}\n'.format(name))
                else:
                    if added_dict[speaker] < threshold:
                        test_file.write('{}\n'.format(name))
                        added_dict[speaker] += 1
                    else:
                        train_file.write('{}\n'.format(name))
                        added_dict[speaker] += 1


main()