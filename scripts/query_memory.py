#!/usr/bin/env python3

import argparse
import json
import psutil
import sys


def main(memory_percentage):
    """
    Computes memory_percentage percent of the available virtual memory.
    Writes the amount of available memory to a dictionary.
    """

    # psutil outputs the memory in bytes. This is converted into
    # mebibytes (1 MiB = 2^20 B) to match CWL's ResourceRequirement input.
    required_memory = int(
        psutil.virtual_memory().available / 2**20 * memory_percentage / 100
    )

    result = {"memory": required_memory}

    with open("./memory.json", "w") as fp:
        json.dump(result, fp)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("memory_percentage", type=int)
    main(parser.parse_args().memory_percentage)
