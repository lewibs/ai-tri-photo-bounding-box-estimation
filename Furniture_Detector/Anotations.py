import pandas as pd
import json
import os
from collections import namedtuple
import math

Anotation = namedtuple("Anotation", ['box', 'label'])

def load_anotations(path):
    df = pd.read_csv(path)

    ids = df.values[:,4]
    metadata = df.values[:,5]

    anotations = {}

    for i, data in enumerate(metadata):
        if type(data) == str:
            data = json.loads(data)
            basename = os.path.basename(ids[i])
            # Split the string using the '-' delimiter
            basename = basename.split('-')

            # Remove the first element from the resulting list
            basename = basename[1:]

            # Join the remaining parts back together using the '-' delimiter
            basename = '-'.join(basename)
            for anotation in data:
                if basename not in anotations:
                    anotations[basename] = []

                anotations[basename].append(Anotation(
                    [anotation["x"], anotation["y"], anotation["x"] + anotation["width"], anotation["y"] + anotation["height"]],
                    anotation["rectanglelabels"]
                ))

    return anotations