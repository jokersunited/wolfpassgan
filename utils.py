import collections
import numpy as np
import re

def tokenize_string(sample):
    return tuple(sample.lower().split(' '))

def load_dataset(path, max_length, tokenize=False, max_vocab_size=2048):
    lines = []

    with open(path, 'r') as f:
        for line in f:
            line = line[:-1]
            if tokenize:
                line = tokenize_string(line)
            else:
                line = tuple(line)

            if len(line) > max_length:
                line = line[:max_length]
                continue # don't include this sample, its too long

            # right pad with ` character
            lines.append(line + ( ("`",)*(max_length-len(line)) ) )

    np.random.shuffle(lines)

    import collections
    counts = collections.Counter(char for line in lines for char in line)

    charmap = {'unk':0}
    inv_charmap = ['unk']

    for char,count in counts.most_common(max_vocab_size-1):
        if char not in charmap:
            charmap[char] = len(inv_charmap)
            inv_charmap.append(char)

    filtered_lines = []
    for line in lines:
        filtered_line = []
        for char in line:
            if char in charmap:
                filtered_line.append(inv_charmap.index(char))
            else:
                filtered_line.append('unk')
        filtered_lines.append(tuple(filtered_line))

    for i in range(100):
        print(filtered_lines[i])

    print("loaded {} lines in dataset".format(len(lines)))
    return filtered_lines, charmap, inv_charmap
