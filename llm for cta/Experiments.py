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

Column 1: TV TV Other TV TV Web
Column 2: 2006 2009 2012 2010 2001 2003 
Column 3: Want to Watch Want to Watch Want to Watch Want to Watch Want to Watch Want to Watch
""", """
Reasoning:
Looking at the table row by row like the first row is "TV," "2006," "Want to Watch," it seems that the first column refers to the type of media, the second column refers to the year associated with each entry, and the third column indicates the status or interest in watching. So, the types would be ["type", "year", "status"].
"""),

    "example2": ("""
Classify these column: 

Column 1: Wednesday Thursday Friday Saturday Sunday Monday
Column 2: Home Home Away Away Home Home 
Column 3: Final Final Final Final Final Final
Column 4: Loss Win Loss Loss Loss Win
""", """
Reasoning:
Looking at the table row by row like the first row is "Wednesday," "Home," "Final," "Loss," it seems that the first column refers to the day of the event, the second column refers to the location, the third column indicates the status of the event, and the fourth column shows the result of the event. So, the types would be ["day", "location", "status", "result"]
"""),

    "example3": ("""
Classify these column: 

Column 1: 24 24 24 24 24 24
Column 2: Los Angeles Los Angeles Los Angeles Los Angeles Los Angeles Los Angeles
Column 3: Hungary Hungary Hungary Hungary Hungary Hungary
Column 4: 19 4 19 9 5 5
""", """
Reasoning:
Looking at the table row by row like the first row is "24," "Los Angeles," "Hungary," "19," it seems that the first column refers to age of a person based on its columns's range, the second column refers to a city, the third column refers to a country, and the fourth column could represent a rank or some form of numeric code. So, the types would be ["code", "city", "country", "rank"]
"""),

    "example4": ("""
Classify these column: 

Column 1: Athina Athina Athina Athina Athina Athina Athina Athina Athina
Column 2: Australia Australia Australia Australia Australia Australia Australia Australia Australia
Column 3: 2 1 2 2 1 1 1 1 2
Column 4: CUB 6, AUS 2 AUS 1, JPN 0 CUB 4, AUS 1 TPE 3, AUS 0 AUS 6, ITA 0 AUS 9, JPN 4 AUS 11, GRE 6 AUS 22, NED 2 CAN 11, AUS 0
""", """
Reasoning:
Looking at the table row by row like the first row is "Athina," "Australia," "2," "CUB 6, AUS 2," it seems that the first column refers to the city where the event took place, the second column refers to a team, the third column indicates the rank of the team based on some criteria, and the fourth column shows the result of a game or match. So, the types would be ["city", "team", "rank", "result"]
""")
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


