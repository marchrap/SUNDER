"""The file that combines utility functions such as txt file generation, etc."""
from tsfresh.feature_extraction import settings
from tsfresh import extract_relevant_features
from tsfresh.feature_extraction import extract_features, EfficientFCParameters
import glob
import nltk
import pandas as pd
import numpy as np
import pickle
import os
import librosa


def create_data_frame(files, channels=False):
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
        channels: A boolean that determines whether channels or speakers are used for determining the sentence units.

    Returns:
        Data frame objects of the form specified above. Each one corresponds to possible channels.
    """
    # Decide on which column to use
    if channels:
        option = 2
    else:
        option = 7

    # Crawl through lines of all the files
    for path in files:
        # Empty dictionary for the channels that collects all their current tokens
        channels = {}

        # Crawl through lines of the given files
        with open(path) as file:
            for line in file:
                # Split the line and create dictionary for the new object
                lexeme = {}
                splitted = line.split()

                # Ignore the comment sections
                if not splitted[0].startswith(";"):
                    # Mark the new channel into the channels dictionary (create separate channels)
                    if splitted[0] == "SPKR-INFO":
                        channels[splitted[option]] = []

                    # Mark the lexeme as the end of sentence if there has been an SU after it
                    elif splitted[0] == "SU":
                        if len(channels[splitted[option]]) > 0:
                            channels[splitted[option]][-1]['end_of_sentence'] = True

                    # Add all the interesting tokens
                    elif splitted[0] == "LEXEME":
                        lexeme['type'] = splitted[0]
                        lexeme['file'] = splitted[1]
                        lexeme['beginning_time'] = float(splitted[3])
                        lexeme['final_time'] = round(float(splitted[3]) + float(splitted[4]), 3)
                        lexeme['token'] = splitted[5]
                        lexeme['subtype'] = splitted[6]
                        lexeme['end_of_sentence'] = False
                        lexeme['pause'] = 0
                        lexeme['turn'] = False

                        # Evaluate pause length by finding the minimum pause between current and all other tokens. Do it
                        # only for the case where the length of the channel is big enough. Also evaluate whether there
                        # is a speaker turn.
                        if len(channels[splitted[option]]) > 0:
                            min_pause = float('inf')
                            previous_speaker = ''
                            for channel, tokens in channels.items():
                                if len(tokens) > 0:
                                    if lexeme['beginning_time'] - tokens[-1]['final_time'] < min_pause:
                                        min_pause = lexeme['beginning_time'] - tokens[-1]['final_time']
                                        previous_speaker = channel

                                        # To eliminate situations in which the pause is negative (due to overlapping) we
                                        # clip it to 0
                                        if min_pause < 0:
                                            min_pause = 0
                                            break

                            # Assign the pause length and speaker turn to the previous token
                            channels[splitted[option]][-1]['pause'] = min_pause
                            channels[splitted[option]][-1]['turn'] = previous_speaker != splitted[option]

                        # Put the new lexeme into the corresponding channel
                        channels[splitted[option]].append(lexeme)

        # Add breaks as the last tokens for all channels
        for channel in channels.keys():
            channels[channel][-1]['end_of_sentence'] = True

        # Create the data frames and return them sorted by time
        combined = []
        for tokens in channels.values():
            combined += tokens
        return pd.DataFrame(combined).sort_values(by='beginning_time').reset_index(drop=True)


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
    start_index = data_frame['beginning_time'].idxmin()
    final_index = data_frame['final_time'].idxmax()

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

    Saves:
        - the data frame with general token, start, end times, etc. ('name'.csv).
        - a numpy file with all the extracted features with pause included (features.npy).
        - the mfcc features (mfcc.npy).
        - the NaN indexes (nan.npy).
        - the y labels (y.npy)

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

            # Obtain the base directory and the final location of files
            base = os.path.splitext(os.path.basename(file))[0]

            # Obtain the pause and turn features for the features
            features['pause'] = df['pause']
            features['turn'] = df['turn']

            # Note the nan indexes and save them
            nan = features.isna().any(1).nonzero()[0]
            np.save(os.path.join(result_path, f'{base}_nan.npy'), nan)

            # Iterate through the rows of the data frame and check the features
            mfcc = []
            for index, row in df.iterrows():
                # Obtain the initial and final samples
                initial = librosa.time_to_samples(row['beginning_time'], sr=sr)
                final = librosa.time_to_samples(row['final_time'], sr=sr)

                # Add the mfcc with variable hop length and y data to the data frame
                hop_length = int((final-initial)/(mfcc_size[1] - 1))
                if hop_length == 0:
                    hop_length = 1
                mfcc.append(librosa.feature.mfcc(y=new_audio[initial:final], n_mfcc=mfcc_size[0], sr=sr,
                                                 hop_length=hop_length).transpose())

            # Save the y labels
            np.save(os.path.join(result_path, f'{base}_y.npy'), y.values)

            # Drop all the NaN rows and save the array n
            np.save(os.path.join(result_path, f'{base}_features.npy'), features.values)

            # Save the mfcc
            np.save(os.path.join(result_path, f'{base}_mfcc.npy'), np.array(mfcc))

            # Save the data frame and print file done
            df.to_csv(os.path.join(result_path, f'{base}.csv'), index=False)
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
    # Obtain the features and filter them. Some of the default features are not very useful but computationally
    # expensive, hence, omitted.
    setting = EfficientFCParameters()
    del setting['number_cwt_peaks']
    del setting['augmented_dickey_fuller']
    del setting['change_quantiles']
    del setting['fft_coefficient']
    del setting['max_langevin_fixed_point']
    del setting['cwt_coefficients']
    features_filtered_direct = extract_relevant_features(data, y, column_id='id', column_sort='t',
                                                         default_fc_parameters=setting)
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

    TODO implement efficient and non efficient versions
    """
    # Open file, audio and calculate relevant fresh features
    files = glob.glob(os.path.join(rttm_path, "*.rttm"))
    files = np.random.choice(files, int(len(files) * percentage), replace=False)

    for file in files:
        try:
            df = create_data_frame((file,))
            new_audio, sr = librosa.load(os.path.splitext(file)[0] + ".wav", sr=None)
            df, y = prepare_data_frame(new_audio, sr, df)
            parameters = calculate_relevant_fresh_features(df, y)
            print(parameters)

            # Save the parameters
            with open(os.path.join(out_path, os.path.splitext(os.path.basename(file))[0] + '_params.dict'), 'wb') as out:
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
