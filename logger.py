import os, sys
import numpy as np
import options
from tqdm import tqdm

opt = options.Options().parse()
max_verbose = opt.verbose

if max_verbose < 4:
    import warnings
    warnings.filterwarnings('ignore')

colors = {
    'plain':    '',
    'H1':       '\033[1m\033[34m',
    'H2':       '\033[37m',
    'H3':       '',
    'end':      '\033[0m',
    'box':      '',
    'debug':    '\033[1m\033[94m',
}

heads = {
    'H1':       '=>',
    'H2':       '  ',
    'H3':       '  \t',
    'plain':    '\t\t',
    'debug':    '\nDEBUG'
}

ends = {
    'H1':       '',
    'H2':       '',
    'H3':       '',
    'plain':    '',
    'debug':    '',
}

def set_box(text, level):
    splits = text.split('\n')
    max_len = len(max(splits, key=lambda p: len(p)))
    indent = heads[level]
    text = '\n'.join([indent + '| ' + s + (' ' * (max_len - len(s))) + ' |' for s in splits])
    text = indent + '=' * (max_len + 4) + '\n' + text + '\n' + indent + '=' * (max_len + 4)

    return text

def print_color(text, level, box=False):
    if box:
        text = set_box(text, level)
    else:
        text = heads[level] + text + ends[level]
    print(colors[level] + text + colors['end'])

def print_text(text, level='plain', verbose=3, box=False):
    if verbose <= max_verbose:
        print_color(text, level, box)

def progress_bar(steps, level='H2', verbose=1):
    if verbose <= max_verbose:
        return tqdm(total=steps, 
                bar_format="{desc}{percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}",
                desc=heads[level])
    else:
        return None

