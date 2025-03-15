import argparse
import sys

import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_triples_file', type=str)
    parser.add_argument('--output_labels_file', type=str)
    parser.add_argument('--selected_relations', type=str, nargs='+')

    args = parser.parse_args()

    df = pd.read_csv(args.input_triples_file)
    filter = df['relation'].isin(args.selected_relations)
    df = df[filter]

    counts = df.groupby('subject')['property_type'].value_counts().unstack()
    valid = (counts.get("Style", 0) == 1) & (counts.get("Genre", 0) == 1)
    invalid_rows = counts[valid == False]

    if not invalid_rows.empty:
        print(f'The following artworks present multiple instances of the selected {args.selected_relations} relations.\n{invalid_rows}')
        sys.exit(1)

    df = df.pivot(index='subject', columns='property_type', values='property').reset_index()

    df.to_csv(args.output_labels_file, index=False)
