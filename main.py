import solver
from helper import *
import program_database
from neural_search import NeuralSearch
from llm import LLM
import json
import os
import traceback
import pandas as pd

correct = 0
total = 0
correct_task_names = []
incorrect_task_names = []

with open("subset.txt", "r") as f:
    files = f.read().splitlines()

program_descriptions = pd.read_csv("./hypo_extraction/program_db_desc.csv")

for task_name in files:
    if task_name in os.listdir("./logs"): continue

    with open(f"ARC-AGI/data/training/{task_name}.json", "r") as f:
        task = json.load(f)

    train_inputs = []
    train_outputs = []
    test_inputs = []
    test_outputs = []

    for ex in task["train"]:
        train_inputs.append(tuple(map(tuple, ex["input"])))
        train_outputs.append(tuple(map(tuple, ex["output"])))

    for ex in task["test"]:
        test_inputs.append(tuple(map(tuple, ex["input"])))
        test_outputs.append(tuple(map(tuple, ex["output"])))

    library = []
    for f_name in dir(solver):
        if f_name[:5] == 'solve' and f_name[6:] not in files:
            description = program_descriptions.loc[(program_descriptions['task_name'] == f_name[6:]+".json")].iloc[0]
            library.append([getattr(solver, f_name), description["output"]])

    program_db = program_database.ProgramDatabase(library)

    neural_search = NeuralSearch()
    try:
        solved = neural_search.search(task_name, train_inputs, train_outputs, test_inputs, test_outputs, program_db)
    except Exception as e:
        print("Error main", e, traceback.format_exc())
        solved = 0
    
    if solved:
        correct+=1
        correct_task_names.append(task_name)
    else:
        incorrect_task_names.append(task_name)

    total += 1
    print(correct/total, correct, total)
    print(correct_task_names, incorrect_task_names, end="\n\n")