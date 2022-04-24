import re
from collections import Counter

def preprocess_sentence(example):
    sent = example['sentence']
    # Make consistent quotes/apostrophes.
    sent = re.sub(r'(?:‘|’|’|“|”|"|”)', "'", sent)
    # Make dashes consistent.
    sent = re.sub(r'(?:—|–)', '-', sent)
    # Many sentences are completely wrapped in quotes and others are not. Remove these.
    if sent.startswith("'") and sent.endswith("'"):
        sent = sent.strip("'")
    example['sentence'] = sent
    return example

def keep_sentence(sent, drop_by_chars=''):
    if sent.isspace():
        return False
    elif any(char in sent for char in drop_by_chars):
        return False
    return True

def get_uncommon_chars(dataset):
    char_counts = Counter(' '.join(dataset['sentence']))
    return [char for char, count in char_counts.items() if count < 295]
