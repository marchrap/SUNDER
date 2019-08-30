"""A file that combines model training and creation."""
from scipy.stats import gamma
import openfst_python as fst
import matplotlib.pyplot as plt
import glob
import os
import math

# The break constant used in modesl
BREAK = "<BREAK>"


def data_parse(files, speakers=True):
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


def create_slm(files_path, plot=True):
    """Creates a sentence length model based on the files passed.

    Args:
        files_path: The path to the *.rttm files that should be parsed in order to obtain their possible lengths.
        plot: A boolean variable that decides whether the results should be plot or not.

    Returns:
        A gamma distribution that represents the obtained model, the maximum sentence length found.
    """
    # Obtain files, their corresponding sentence lengths and the maximum sentence length seen
    files = glob.glob(os.path.join(files_path, "*.rttm"))
    lengths = data_parse(files)
    max_seen = max(lengths)

    # Fit in the gamma distribution
    shape, loc, scale = gamma.fit(lengths, floc=0)
    gamma_generator = gamma(shape, loc, scale)

    if plot:
        fig, ax = plt.subplots(1, 1)
        ax.hist(lengths, bins=max(lengths), label='samples')
        x_values = range(max(lengths) + 1)
        ax.plot(x_values, gamma_generator.pdf(x_values) * len(lengths), 'r-', lw=2, label='gamma distribution')
        ax.legend(loc="best", frameon=False)
        plt.show()

    return gamma_generator, max_seen


def create_fst_slm(model, tokens, max_seen):
    """Creates an fst representation of the sentence language model based on the tokens provided.

    Args:
         model: A gamma generator that allows for obtaining the cdf of the model.
         tokens: A list of tokens that are within a file.
         max_seen: The maximum sentence length seen in training.

    Returns:
        An FST graph that represents the sentence length model.

    TODO ask about correctness of assumption that composition would leave it as all with the same probability.
    TODO ask whether I should output the models as FST and then compose or do it at one stage from models.
    """
    # Initialise a new, symbol table
    symbol_table = fst.SymbolTable()
    for c in ['<eps>', BREAK] + tokens:
        symbol_table.add_symbol(c)

    # Initialise the compiler
    compiler_zero = fst.Compiler(isymbols=symbol_table, osymbols=symbol_table, keep_isymbols=True, keep_osymbols=True)
    compiler_non_zero = fst.Compiler(isymbols=symbol_table, osymbols=symbol_table, keep_isymbols=True,
                                     keep_osymbols=True)

    # Combine the tokens into a set
    tokens = set(tokens)

    # Deal with all the words
    for x in range(max_seen + 1):
        if x < max_seen:
            # We do not add the probability of the next word for the final state as there cannot be any next word. The
            # important bit here is that words do not actually possess any probability. They do not have any cost. We
            # cannot treat the situation as a probability tree as the events are not independent. Thus, we sort of fit
            # this distribution to the FST by only weighting breaks! Look into the notebook for more information.
            for token in tokens:
                print(f'{x} {x + 1} {token} {token} 0', file=compiler_non_zero)
                print(f'{x} {x + 1} {token} {token} 0', file=compiler_zero)

        if x != 0:
            # We do not add the probability of a break for word of zero length
            # Also we approximate the word to be spanning from x-0.5 to x+0.5
            break_probability = model.cdf(x + 0.5) - model.cdf(x - 0.5)
            break_score = - math.log(break_probability)
            print(f'{x} 0 {BREAK} {BREAK} {break_score}', file=compiler_non_zero)
            print(f'{x} 0 {BREAK} {BREAK} 0', file=compiler_zero)

    # Print finish state and return the graph
    print('0', file=compiler_non_zero)
    print('0', file=compiler_zero)
    return compiler_non_zero.compile(), compiler_zero.compile()


def create_fst_prosodic(model, scaled_features, tokens):
    """Creates and fst model for the prosodic model.

    Args:
        model: A keras model that can be called with predict.
        scaled_features: A numpy array of features that can be used to predict the model output.
        toknes: A list of tokens in chronological order that have occurred in the analysed file.

    Returns:
         A FST graph that represents the prosodic model.
    """
    # Ensure the length of the scaled features is the same as of the token
    assert len(scaled_features) == len(tokens), "The tokens and scaled features must be of the same length!"

    # Initialise a new, symbol table
    symbol_table = fst.SymbolTable()
    for symbol in ['<eps>', BREAK] + tokens:
        symbol_table.add_symbol(symbol)

    # Initialise the compiler
    compiler_zero = fst.Compiler(isymbols=symbol_table, osymbols=symbol_table, keep_isymbols=True, keep_osymbols=True)
    compiler_non_zero = fst.Compiler(isymbols=symbol_table, osymbols=symbol_table, keep_isymbols=True,
                                     keep_osymbols=True)

    # Predict the labels
    labels = scaled_features#model.predict(scaled_features)

    # Add them to the FST compilers
    for index, token in enumerate(tokens):
        # Add the normal one
        print(f'{2 * index} {2 * index + 1} {token} {token} 0', file=compiler_non_zero)
        print(f'{2 * index + 1} {2 * index + 2} {BREAK} {BREAK} {- math.log(labels[index])}', file=compiler_non_zero)
        print(f'{2 * index + 1} {2 * index + 2} <eps> <eps> {- math.log(1 - labels[index])}', file=compiler_non_zero)

        # Add the zero one
        print(f'{2 * index} {2 * index + 1} {token} {token} 0', file=compiler_zero)
        print(f'{2 * index + 1} {2 * index + 2} {BREAK} {BREAK} 0', file=compiler_zero)
        print(f'{2 * index + 1} {2 * index + 2} <eps> <eps> 0', file=compiler_zero)

    # Add the final state to the FST and return the ready graph
    print(f'{2 * len(tokens)}', file=compiler_zero)
    print(f'{2 * len(tokens)}', file=compiler_non_zero)
    return compiler_non_zero.compile(), compiler_zero.compile()


def create_fsts(file_path, train_rttm_path, model_path, feature_scaler_path, mfcc_scaler_path):
    """A general function that allows to create the fsts for a given file.

    Args:
        file_path: The path to the files to be analysed.
        train_rttm_path: The path to the training rttm files.
        model_path: The path to the model.
        feature_scaler_path: The path to the feature scaler.
        mfcc_scaler_path: The path to the mfcc scaler.

    TODO change the raw file parsing to data frames!
    """
    pass


if __name__ == "__main__":
    files = glob.glob('data/*.rttm')
    model, max_seen, tokens = create_slm("data")
    create_fst_slm(model, tokens, max_seen)
