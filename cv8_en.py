import re
from collections import Counter
from datasets import load_dataset
from datasets.features.audio import Audio
from functools import partial

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

def prepare(split, use_pct, sampling_rate=16_000, seed=1, uncommon_chars=None):
    ds = load_dataset(
        "mozilla-foundation/common_voice_8_0", "en", split=f'{split}[:{int(use_pct*100)}%]', use_auth_token=True
    )
    ds = ds.cast_column('audio', Audio(sampling_rate=sampling_rate)).map(preprocess_sentence)
    if uncommon_chars is None:
        uncommon_chars = get_uncommon_chars(ds)
    print(f'Dropping sentences that contain any of these characters:\n{uncommon_chars}')
    ds = ds.filter(partial(keep_sentence, drop_by_chars=uncommon_chars), input_columns='sentence')
    # return ds.shuffle(SEED).select(range(int(use_pct * ds.num_rows))), uncommon_chars
    return ds.shuffle(seed), uncommon_chars