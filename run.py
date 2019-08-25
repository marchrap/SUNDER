from preprocessing import create_data_frame, calculate_relevant_fresh_features
import pickle
import librosa
import glob
import os
import numpy as np

# Open file, audio and calculate relevant fresh features
files = glob.glob("data/train/switchboard/*.rttm")
files = np.random.choice(files, int(len(files) * 0.15), replace=False)

for file in files:
    try:
        df = create_data_frame((file,))
        newAudio, sr = librosa.load(os.path.splitext(file)[0] + ".wav", sr=None)
        parameters = calculate_relevant_fresh_features(newAudio, sr, df)
        print(parameters)

        # Save the parameters
        with open(os.path.splitext("preprocessing/switchboard/" + os.path.basename(file))[0] + '_params.dict', 'wb') as file:
            pickle.dump(parameters, file, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(e)

dicts = glob.glob('preprocessing/switchboard/*')
fc_features = intersect_dictionaries(dicts, 0.5)
files = glob.glob("data/train/switchboard/*.rttm")

for file in files[:1]:
    try:
        df = create_data_frame((file,))
        new_audio, sr = librosa.load(os.path.splitext(file)[0] + ".wav", sr=None)
        data, y = prepare_data_frame(new_audio, sr, df)
        features = extract_fresh_features(data, fc_features)
        base = os.path.splitext(os.path.basename(file))[0]
        location = os.path.splitext(file)[0]
        os.mkdir(location)
        for index, row in data_frame.iterrows():
            current = {}
            initial = librosa.time_to_samples(row['beginning_time'], sr=sr)
            final = librosa.time_to_samples(row['final_time'], sr=sr)
            current['mfcc'] = librosa.feature.mfcc(y=new_audio[intial:final])
            current['y'] = y.iloc[index]
            current['raw_features'] = np.array(features.loc[features.id == index].tolist())
            with open(f'{location}/{index}.data', 'rb') as file:
                pickle.dump(current, file, protocol=pickle.HIGHEST_PROTOCOL)
        df.to_csv(f'{location}/{base}.csv')
    except Exception as e:
        print(e)
