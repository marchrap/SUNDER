"""A file that combines model training and creation."""
from scipy.stats import gamma
import numpy as np
import matplotlib.pyplot as plt
import glob
import os


def sentence_length_parse(files, speakers=True):
    """Parses the data and returns a list with the sentence lengths found.

    The parsing analysis is conducted on the speaker or channel basis dependent on speakers argument.

    Args:
        files: A list of files to be parsed.
        speakers: A boolean value that decides whether the parsing is conducted on the speakers/channel basis.

    Returns:
        A list in the form [length, length, ...] that repeats the length of sentences found by the number of occurances
        of the given length.
    """
    # Decide on the channel/speaker analysis basis
    if speakers:
        option = 7
    else:
        option = 2

    # Crawl through all the files and results to the corresponding list
    results = []
    for file in files:
        channels = {}

        # Crawl through lines of the given files
        with open(file) as output:
            for line in output:
                # Split the line and create dictionary for the new object
                splitted = line.split()

                # Ignore the comment sections
                if not splitted[0].startswith(";"):
                    # Mark the new channel into the channels dictionary (create separate channels)
                    if splitted[0] == "SPKR-INFO":
                        channels[splitted[option]] = 0

                    # Add one to the results dictionary and reset current channel
                    elif splitted[0] == "SU":
                        if channels[splitted[option]] > 0:
                            results.append(channels[splitted[option]])
                            channels[splitted[option]] = 0

                    # Add one to the corresponding channel
                    elif splitted[0] == "LEXEME":
                        channels[splitted[option]] += 1

            # Add the last unprocessed batches to the corresponding result slots
            for channel in channels.keys():
                if channels[channel] > 0:
                    results.append(channels[channel])

    return results


def token_parse(files):
    """Parses the data and returns a set of tokens found.

    Args:
        files: A list of files to be parsed.

    Returns:
        A token set found.
    """
    # Crawl through all the files and results to the corresponding set
    tokens = set()
    for file in files:
        with open(file) as output:
            for line in output:
                # Split the line and create dictionary for the new object
                splitted = line.split()

                # Ignore the comment sections
                if not splitted[0].startswith(";"):
                    if splitted[0] == "LEXEME":
                        tokens.add(splitted[5])

    return tokens


def create_slm(files_path, plot=True):
    """Creates a sentence length model based on the files passed.

    Args:
        files_path: The path to the *.rttm files that should be parsed in order to obtain their possible lengths.
        plot: A boolean variable that decides whether the results should be plot or not.

    Returns:
        A gamma distribution that represents the obtained model and the maximum sentence length found.
    """
    # Obtain files, their corresponding sentence lengths and the maximum sentence length seen
    files = glob.glob(os.path.join(files_path, "*.rttm"))
    lengths = sentence_length_parse(files)
    max_seen = max(lengths)

    # Fit in the gamma distribution
    shape, loc, scale = gamma.fit(lengths, floc=0)
    gamma_generator = gamma(shape, loc, scale)

    if plot:
        fig, ax = plt.subplots(1, 1)
        ax.hist(lengths, bins=max(lengths), label='samples')
        x_values = range(0, max(lengths) + 1)
        ax.plot(x_values, gamma_generator.pdf(x_values) * len(lengths), 'r-', lw=2, label='gamma distribution')
        ax.legend(loc="best", frameon=False)
        plt.show()

    return gamma_generator, max_seen


if __name__ == "__main__":
    files = glob.glob('data/*.rttm')
    a, max_seen = create_slm("data")
    print(a.pdf(0))
    tokens = token_parse(files)
