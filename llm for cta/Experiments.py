import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import json
from collections import Counter
from collections import defaultdict
from groq import Groq
import openai
import ast
from numpy.linalg import norm
import re
import matplotlib.pyplot as plt

from Prompting import result

openai.api_key = "YOUR_API_KEY"

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)

    norm_a = norm(vector_a)
    norm_b = norm(vector_b)

    similarity = dot_product / (norm_a * norm_b)
    return similarity

#reading dataset sample and training set sample for RAG
final_dataset = pd.read_csv("final_dataset.csv")
RAG_2000_sample = pd.read_csv("RAG_2000_sample.csv")

# Few-shot examples dictionary
single_column_examples = {
    "example1": ("Column 1: Alabama Alaska Arizona Arkansas California Colorado Connecticut Delaware District of Columbia Florida Georgia Hawaii Idaho", "state"),
    "example2": ("Column 1: DJ REVOLUTION DJ REVOLUTION KANYE WEST ATMOSPHERE JAY-Z", "artist"),
    "example3": ("Column 1: South Orange Orlando Jupiter San Francisco San Francisco San Francisco San Francisco New York", "city"),
    "example4": ("Column 1: Alabama Florida State Ohio State Baylor Stanford Oregon Clemson Auburn Texas A&M Oklahoma St. Missouri South Carolina", "team"),
}

def single_column_prompting(dataset, few_shot=True):
    results = []
    cnt = 0

    for i in range(len(dataset)):
        list_temp = dataset['data'][i][0:250]

        messages = [
            {
                "role": "system",
                "content": (
                    "Your task is to classify the columns of a given table with only one of the following classes that are seperated with comma:"
                    "sex, category, album, status, origin, format, day, location, notes, duration, nationality, region, club, address, rank, name, "
                    "position, description, country, state, city, code, symbol, isbn, age, type, gender, team, year, company, result, artist."
                ),
            },
            {
                "role": "system",
                "content": (
                    "Your instructions are: 1. Look at the column and the types given to you. 2. Examine the values of the column. "
                    "3. Select a type that best represents the meaning of the column. 4. Answer with the selected type only, and print the type only once. "
                    "The format of the answer should be like this: type  Print 'I don't know' if you are not able to find the semantic type."
                ),
            },
        ]

        # Conditionally add few-shot examples
        if few_shot:
            for _, (example_input, example_output) in single_column_examples.items():
                messages.append({"role": "user", "content": f"Classify this column:\n\n{example_input}"})
                messages.append({"role": "assistant", "content": example_output})

        # Add actual input
        messages.append({"role": "user", "content": f"Classify this column:\n\n{list_temp}"})

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )
        results.append(response['choices'][0]["message"]["content"].strip())

        cnt += 1
        if cnt % 10 == 0:
            print(cnt)

    return results



def RAG_prompting(test_set, train_set):
    from ast import literal_eval

    correct_count = 0
    results = []

    for i in range(50, len(test_set)):
        query_embedding = get_embedding(test_set['data'][i])

        # Initialize with low scores and placeholders
        few_shot_tuples = [(-float('inf'), "class", "data")] * 4

        for j in range(len(train_set)):
            candidate_embedding = literal_eval(train_set["embedding"][j])
            similarity = cosine_similarity(query_embedding, candidate_embedding)

            # Find the lowest scoring example to potentially replace
            min_score = min(few_shot_tuples, key=lambda x: x[0])
            min_index = few_shot_tuples.index(min_score)

            if similarity > min_score[0]:
                few_shot_tuples[min_index] = (similarity, train_set["class"][j], train_set["data"][j])

        # Prepare prompt parts
        examples = [
            {"user": f"Column 1: {t[2][:500]}", "assistant": t[1]}
            for t in sorted(few_shot_tuples, key=lambda x: -x[0])
        ]
        user_input = {"role": "user", "content": f"Column 1: {test_set['data'][i][:500]}"}

        # Quick correctness check
        gt_class = test_set["class"][i]
        if any(gt_class == example["assistant"] for example in examples):
            correct_count += 1

        # Build message sequence
        messages = [
            {
                "role": "system",
                "content": (
                    "Your task is to classify the columns of a given table with only one of the following classes "
                    "that are separated with comma: sex, category, album, status, origin, format, day, location, notes, "
                    "duration, nationality, region, club, address, rank, name, position, description, country, state, city, "
                    "code, symbol, isbn, age, type, gender, team, year, company, result, artist."
                ),
            },
            {
                "role": "system",
                "content": (
                    "Your instructions are: 1. Look at the column and the types given to you. "
                    "2. Examine the values of the column. 3. Select a type that best represents the meaning of the column. "
                    "4. Answer with the selected type only, and print the type only once. "
                    "The format of the answer should be like this: type  Print 'I don't know' if you are not able to find the semantic type."
                ),
            },
        ]

        for ex in examples:
            messages.append({"role": "user", "content": ex["user"]})
            messages.append({"role": "assistant", "content": ex["assistant"]})
        messages.append(user_input)

        # Query GPT
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )
        results.append(response['choices'][0]['message']['content'].strip())

        if i % 10 == 0:
            print(f"Processed: {i}")

    return results


# Few-shot examples for multi-column classification
multi_column_examples = {
    "example1": ("""
Classify these column: 

Column 1: Away Away Away Away Away Away Away Home
Column 2: Final Final Final Final Final Final Final Final
Column 3: Loss Loss Loss Win Win Loss Win Win
""", "['location', 'status', 'result']"),

    "example2": ("""
Classify these column: 

Column 1: Jeff Fink nathaniel wells Brook Bielen Ken Hahn Andrew Jansy michael thoen Andrew Hayes Nick Graves Michael Jones Mark Deresta Lloyd Connelly Andy Crump Ricardo Medina Justin Huggins Erik Denninghoff Marcelo Heredia Adam Keir Dante Solano Reynaldo Ortiz rudy vega Mark Vickers
Column 2: Dimwits Rogue / Mt. View Cycles Wolverine Mountain View Cycles/Subway DNR Cycling Rogue / Mt. View Cycles DNR Cycling Camas Bike and Sport DNR Cycling Upper Echelon Fitness and Rehabilitation
Column 3: White Salmon Mosier Portland Portland Portland White Salmon Hood River Portland Hood River Portland Oregon City Vancouver Portland Portland Vancouver Portland Portland Portland Forest Grove
""", "['name', 'team', 'city']"),

    "example3": ("""
Classify these column: 

Column 1: Greenhouse Gases Greenhouse Gases Greenhouse Gases Greenhouse Gases Greenhouse Gases
Column 2: C13/C12 in Carbon Dioxide (d13C (CO2)) C13/C12 in Carbon Dioxide (d13C (CO2)) C13/C12 in Carbon Dioxide (d13C (CO2)) C13/C12 in Carbon Dioxide (d13C (CO2)) C13/C12 in Carbon Dioxide
Column 3: Flask Flask Flask Flask Flask
""", "['category', 'name','type']"),

    "example4": ("""
Classify these column: 

column 1: Introduction to Paddle Boarding Intro to Paddle Boar Intro to Paddle Boar
column 2: Phillips Lake Park Phillips Lake Park Phillips Lake Park
column 3: Unavailable Unavailable Unavailable
""", "['description', 'location', 'status']")
}


def multi_column_prompt_gpt(dataset, few_shot=True):
    results = []
    table_ids = dataset['table_id'].unique()

    for cnt, table_id in enumerate(table_ids):
        # Aggregate column data for a given table
        filtered_rows = dataset[dataset['table_id'] == table_id]['data'].iloc[:200]
        list_temp = "\n".join([f"column {i + 1}: {val}" for i, val in enumerate(filtered_rows)])

        # Build the base system prompt
        messages = [
            {
                "role": "system",
                "content": (
                    "Your task is to classify the columns of a given table with only one of the following classes that are seperated with comma:"
                    " sex, category, album, status, origin, format, day, location, notes, duration, nationality, region, club, address, rank, name, "
                    "position, description, country, state, city, code, symbol, isbn, age, type, gender, team, year, company, result, artist."
                )
            },
            {
                "role": "system",
                "content": (
                    "Your instructions are: 1. Look at the columns and the types given to you. 2. Examine the values of the columns. "
                    "3. Select a type that best represents the meaning of each column. 4. Answer with the selected type only. "
                    "The format of the answer should be like this: ['type1', 'type2', 'type3']. Print 'I don't know' if you are not able to find the semantic type."
                )
            },
        ]

        # Add few-shot examples if enabled
        if few_shot:
            for example, answer in multi_column_examples.values():
                messages.append({"role": "user", "content": example})
                messages.append({"role": "assistant", "content": answer})

        # Add the current table to be classified
        messages.append({"role": "user", "content": list_temp})

        # Query GPT
        openai.api_key = "Your API Key"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        results.append(response['choices'][0]["message"]["content"].strip())

        if cnt % 10 == 0:
            print(cnt)

    return results


