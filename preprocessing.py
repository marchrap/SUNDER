"""The file that combines utility functions such as txt file generation, etc."""
from tsfresh.feature_extraction import settings
from tsfresh import extract_relevant_features
from tsfresh.feature_extraction import extract_features
import glob
import nltk
import pandas as pd
import numpy as np
import pickle
import os
import librosa


def create_data_frame(files):
    """Creates a pandas DataFrame from the provided RTTM files.

    Data frame is of the form:
        - beginning_time
        - final_time
        - file
        - end_of_sentence
        - pause
        - token

    Args:
        files: A list of files that should be parsed to obtain the data frame.

    Returns:
        A data frame object of the form specified above.
    """
    # Crawl through lines of all the files
    for path in files:
        # Empty dictionary for the speakers that collects all their current tokens
        speakers = {}

        # Crawl through lines of the given files
        with open(path) as file:
            for line in file:
                # Split the line and create dictionary for the new object
                lexeme = {}
                splitted = line.split()

                # Ignore the comment sections
                if not splitted[0].startswith(";"):
                    # Mark the new speaker into the speakers dictionary (create separate channels)
                    if splitted[0] == "SPKR-INFO":
                        speakers[splitted[7]] = []

                    # Mark the lexeme as the end of sentence if there has been an SU after it
                    elif splitted[0] == "SU":
                        if len(speakers[splitted[7]]) > 0:
                            speakers[splitted[7]][-1]['end_of_sentence'] = True

                    # Add all the interesting tokens
                    elif splitted[0] == "LEXEME":
                        lexeme['type'] = splitted[0]
                        lexeme['file'] = splitted[1]
                        lexeme['beginning_time'] = float(splitted[3])
                        lexeme['final_time'] = round(float(splitted[3]) + float(splitted[4]), 3)
                        lexeme['token'] = splitted[5]
                        lexeme['subtype'] = splitted[6]
                        lexeme['end_of_sentence'] = False

                        # Evaluate pause length by finding the minimum pause between current and all other tokens
                        min_pause = float('inf')
                        for tokens in speakers.values():
                            if len(tokens) > 0:
                                if lexeme['beginning_time'] - tokens[-1]['final_time'] < min_pause:
                                    min_pause = lexeme['beginning_time'] - tokens[-1]['final_time']

                                    # To eliminate situations in which the pause is negative (due to overlapping) we
                                    # clip it to 0
                                    if min_pause < 0:
                                        min_pause = 0
                                        break

                        # Add pause length and put the new lexeme into the corresponding speaker's channel
                        lexeme['pause'] = min_pause
                        speakers[splitted[7]].append(lexeme)

        # Add breaks as the last tokens for all speakers
        for speaker in speakers.keys():
            speakers[speaker][-1]['end_of_sentence'] = True

        # Combine the arrays and create a data frame
        combined = []
        for tokens in speakers.values():
            combined += tokens
        return pd.DataFrame(combined)


def prepare_data_frame(audio_array, sampling_rate, data_frame):
    """Prepares the data frame for feature extraction.

    Args:
        audio_array: The numpy array as loaded by librosa
        sampling_rate: The sampling rate as provided by librosa
        data_frame: The pandas data frame that contains the previously parsed words.

    Returns:
        A pandas data frame object that tsfresh can extract features from and an array of which id is a failed one.
    """
    # Obtain the y labels
    y = data_frame['end_of_sentence']

    # Obtain the start and end indexes
    start_index = data_frame.loc[:, 'beginning_time'].idxmin()
    final_index = data_frame.loc[:, 'final_time'].idxmax()

    # Cut the audio correspondingly
    start = librosa.time_to_samples(data_frame.loc[start_index, 'beginning_time'], sr=sampling_rate)
    end = librosa.time_to_samples(data_frame.loc[final_index, 'final_time'], sr=sampling_rate)
    audio = audio_array[start:end + 1]

    # Create the necessary rows
    data = []
    for index, row in data_frame.iterrows():
        initial = librosa.time_to_samples(row['beginning_time'], sr=sampling_rate) - start
        final = librosa.time_to_samples(row['final_time'], sr=sampling_rate) - start
        data += ((index, time, audio[time]) for time in range(initial, final))

    # Convert the rows into a data frame necessary for TSFRESH
    data = pd.DataFrame(data, columns=['id', 't', 'x'])

    return data, y


def extract_labels(dictionary_path, file_path, result_path, mfcc_size):
    """Extracts the necessary labels (x, y) for the training of the network.

    Saves the data frame with general token, start, end times, etc. and with all the extracted features (together with
        mfcc and whether there are any NaN's column).

    Args:
        dictionary_path: A path to dictionaries that have some features and should be intersected.
        file_path: Path to the rttm files to be analysed.
        result_path: Path to the place where the results will be stored.
        mfcc_size: A tuple of shape 2, that specifies the size of the mfcc filter.
    """
    # Extract files and dictionaries
    dicts = glob.glob(os.path.join(dictionary_path, '*'))
    files = glob.glob(os.path.join(file_path, "*.rttm"))

    # Extract the fresh relevant features by majority votes
    fc_features = intersect_dictionaries(dicts, 0.5)

    # Crawl through the files
    for base_index in range(len(files)):
        # For printing the current file number we use index based for loop
        file = files[base_index]

        # In case there is an error we just pass the current iteration
        try:
            # Create data frame, extract audio and features
            df = create_data_frame((file,))
            new_audio, sr = librosa.load(os.path.splitext(file)[0] + ".wav", sr=None)
            data, y = prepare_data_frame(new_audio, sr, df)
            features = extract_fresh_features(data, fc_features)

            # Obtain the base directory
            base = os.path.splitext(os.path.basename(file))[0]

            # Merge the data frame and features data frames
            df = pd.merge(df, features, left_index=True, right_index=True)

            # Note the nan indexes and add them to the data frame
            df['NaN'] = df.isna().any(1)

            # Add the mfcc column to the data frame
            df['mfcc'] = None

            # Iterate through the rows of the data frame and check the features
            for index, row in df.iterrows():
                # Obtain the initial and final samples
                initial = librosa.time_to_samples(row['beginning_time'], sr=sr)
                final = librosa.time_to_samples(row['final_time'], sr=sr)

                # Add the mfcc with variable hop length and y data to the data frame
                features.at[index, 'mfcc'] = librosa.feature.mfcc(y=new_audio[initial:final], n_mfcc=mfcc_size[0],
                                                                  hop_length=int((initial-final)/(mfcc_size[1] - 1))).transpose()

            df.to_csv(os.path.join(result_path, f'{base}.csv'))
            print(f'Done file number {base_index} - {base}')
        except Exception as e:
            print(e)


def calculate_relevant_fresh_features(data, y):
    """Calls the TSFRESH code and extracts the relevant kind list of features from the given audio_array.

    Args:
        data: A pandas data frame object that contains the prepared words
        y: Labels with id.

    Returns:
        A pandas data frame object that contains the extracted features for each id of the previously provided data
        frame.
    """
    # Obtain the features and filter them
    features_filtered_direct = extract_relevant_features(data, y, column_id='id', column_sort='t')
    print(features_filtered_direct)
    return settings.from_columns(features_filtered_direct)


def extract_fresh_features(data, fc_parameters):
    """Extracts the features based on the data provided.

    Args:
        data: Pandas data frame that contains prepared words.
        fc_parameters: A dictionary specifying which features should be extracted.

    Returns:
        The extracted features.
    """
    return extract_features(data, column_id="id", column_sort="t", default_fc_parameters=fc_parameters)


def obtain_feature_dictionaries(rttm_path, out_path, percentage=1, efficient=True):
    """Obtains the sampled dictionaries from the path provided.

    Args:
        rttm_path: The path to the location of rttm files to be analysed. The corresponding wav files should also be
            located there.
        out_path: The path to which the relevant feature dictionaries are going to be output.
        percentage: The percentage of files that will be sampled from the data. A float between 1 and 0.
        efficient: A boolean that specifies whether the features used should be efficient ones or not.

    TODO think about doing join instead of intersection
    TODO implement efficient and non efficient versions
    """
    # Open file, audio and calculate relevant fresh features
    files = glob.glob(rttm_path + "*.rttm")
    files = np.random.choice(files, int(len(files) * percentage), replace=False)

    for file in files:
        try:
            df = create_data_frame((file,))
            new_audio, sr = librosa.load(os.path.splitext(file)[0] + ".wav", sr=None)
            df, y = prepare_data_frame(new_audio, sr, df)
            parameters = calculate_relevant_fresh_features(df, y)
            print(parameters)

            # Save the parameters
            with open(os.path.splitext(out_path + os.path.basename(file))[0] + '_params.dict', 'wb') as out:
                pickle.dump(parameters, out, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(e)


def intersect_dictionaries(dicts, threshold=1.0):
    """Conducts a feature vote process.

    The features that occur in the data threshold percent of the time are left. Others are rejected.

    Args:
        dicts: The list of dictionaries that should be intersected.
        threshold: A float parameter between 1 and 0 that allows to set the threshold of the voting process.

    Returns:
        dictionary that contains only the intersection of the dictionaries.

    TODO Consider more features instead of only small subset.
    """
    # Extract dictionaries from files provided
    dictionaries = []
    for file in dicts:
        with open(file, 'rb') as dictionary:
            dictionaries.append(pickle.load(dictionary))

    # Define the occurrence dictionary and process the dictionaries for the occurrences
    occurrence = {}
    for dictionary in dictionaries:
        for key in dictionary['x'].keys():
            if key in occurrence:
                occurrence[key] += 1
            else:
                occurrence[key] = 1

    # Define the intersection object that is created by majority vote
    intersection = set()
    for key in occurrence.keys():
        # Conduct the vote and ignore all the frequency features as for those we have MFCC
        if occurrence[key] >= int(len(dictionaries) * threshold) and "fft" not in key:
            intersection.add(key)

    # Create final dictionary by composing all previous ones
    final_dictionary = {key: [] for key in intersection}
    for dictionary in dictionaries:
        for key in dictionary['x'].keys():
            if key in intersection:
                if dictionary['x'][key]:
                    final_dictionary[key] += [element for element in dictionary['x'][key] if element not in
                                              final_dictionary[key]]
                else:
                    final_dictionary[key] = None

    # Calculate total number of features
    features = 0
    for key in final_dictionary.keys():
        if final_dictionary[key]:
            features += len(final_dictionary[key])
        else:
            features += 1

    # Print total number of features and return the dictionary
    print(f'Total number of features: {features}')
    return final_dictionary


def generate_txt(file_list, merge=False):
    """Generates a txt file of tokenized words out of the file names provided.

    Files are output to the data folder.

    Args:
        file_list: A tuple of file locations that can be read and are in the rttm format specified by LDC.
        merge: A boolean value that specifies whether the output should be merged into one file.
    """
    # Crawl through the files and for each generate a list of sentences
    sentences = []
    for input_file in file_list:
        current_sentence = ""

        # Eliminates the first empty sentence
        is_first = True

        with open(input_file) as file:
            for line in file:
                splitted = line.split()
                if splitted[0] == "LEXEME":
                    current_sentence += splitted[5] + " "
                elif splitted[0] == "SU":
                    if not is_first:
                        sentences.append(current_sentence)
                        current_sentence = ""
                    else:
                        is_first = False

            # Append the last unended sentence
            sentences.append(current_sentence)

        if not merge:
            with open(f"data/{input_file.split('.')[0]}.txt", 'w'):
                for sentence in sentences:
                    file.write(' '.join(nltk.word_tokenize(sentence)).lower() + '\n')

            sentences = []

    if merge:
        with open(f"data/merged_txt.txt", 'w'):
            for sentence in sentences:
                file.write(' '.join(nltk.word_tokenize(sentence)).lower() + '\n')

    return True
