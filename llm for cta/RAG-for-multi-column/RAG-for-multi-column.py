import pandas as pd
import openai
import numpy as np
from numpy.linalg import norm
import ast


openai.api_key = "YOUR_API_KEY"


traiset_for_multi_rag = pd.read_csv("trainset_for_multi_rag.csv") # This is the training set for the RAG model
trainset_for_multi_rag_embeds = pd.read_csv("trainset_for_multi_rag_embeds.csv") # This is the serialized version of the training set tables among with embedding for the serialized tables
final_dataset = pd.read_csv("final_dataset.csv") #This is the test set

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

def cosine_similarity(vector_a, vector_b):
    # Calculate the dot product of the two vectors
    dot_product = np.dot(vector_a, vector_b)

    # Calculate the L2 (Euclidean) norms of the vectors
    norm_a = norm(vector_a)
    norm_b = norm(vector_b)

    # Calculate the cosine similarity
    similarity = dot_product / (norm_a * norm_b)

    return similarity

# function for making each tables a serialized string and separating columns with | to implement RAG on the test tables
def format_table(group):
    columns = []
    for idx, sub_group in group.groupby('col_idx'):
        col_string = f"Column {idx}: " + " ".join(sub_group['data'][0:200].tolist())
        columns.append(col_string)
    return " | ".join(columns)


# creates the correct formatting of few shot examples with the desired output
def example_creator(group):
    data_string = "Classify these columns:\n\n"
    class_list = []

    for idx in sorted(group['col_idx'].unique()):
        column_data = group[group['col_idx'] == idx]['data'].values[0]
        column_class = group[group['col_idx'] == idx]['class'].values[0]
        data_string += f"Column {idx}: {column_data}\n"
        class_list.append(column_class)

    class_string = f'\n{class_list}'

    return data_string, class_string


def RAG_prompting_multiـcolumn(test_set, training_set_with_embedding, original_training_set):
    cntt = 0
    column_count = 0
    grouped = original_training_set.groupby('table_id')
    results = grouped.apply(example_creator)
    grouped_test = test_set.groupby('table_id')
    results_test = grouped_test.apply(example_creator)

    output = []
    checked_ids = []

    for i in range(len(test_set)):
        if test_set["table_id"][i] in checked_ids:
            continue
        else:

            # Step 1: Create the formatted string for the current table in the test set
            current_table = test_set[test_set['table_id'] == test_set['table_id'].iloc[i]]
            formatted_test_string = format_table(current_table)

            # Step 2: Get the embedding for the formatted string
            list1 = get_embedding(formatted_test_string)  # Ensure it's a 1D array

            # Step 3: Initialize few-shot tuples with the lowest cosine similarities
            few_shot_tuples = [(-float('inf'), "class", "data", "table_id") for _ in range(4)]
            min_index = None

            # Step 4: Find the most similar strings in the train set
            for j in range(len(training_set_with_embedding)):
                list2 = ast.literal_eval(training_set_with_embedding["embedding"][j])
                cos_sim = cosine_similarity(list1, list2)  # Cosine similarity between two 1x1536 arrays

                min_value = min(few_shot_tuples, key=lambda x: x[0])[0]

                if cos_sim > min_value:
                    min_index = few_shot_tuples.index(min(few_shot_tuples, key=lambda x: x[0]))
                    data_string, class_string = results[training_set_with_embedding["table_id"][j]]
                    few_shot_tuples[min_index] = (cos_sim, class_string, data_string, training_set_with_embedding["table_id"][j])

            # Step 5: Prepare examples from the original DataFrame
            examples = []
            answers = []
            for index, (_, class_label, _, table_id) in enumerate(few_shot_tuples):
                similar_table = original_training_set[original_training_set['table_id'] == table_id]
                example = f"Classify these column:\n"
                for col_idx in similar_table['col_idx'].unique():
                    col_data = " ".join(similar_table[similar_table['col_idx'] == col_idx]['data'].tolist())
                    example += f"Column {col_idx}: {col_data}\n"
                answer = f'["{class_label}"]'
                examples.append(f'example{index + 1}_multi_reason = """\n{example}"""')
                answers.append(f'answer{index + 1}_multi_reason = """\n{answer}"""')
            data_string_test, class_string_test = results_test[test_set["table_id"][i]]
            if set(class_string_test) == set(few_shot_tuples[0][1]) or set(class_string_test) == set(
                    few_shot_tuples[1][1]) or set(class_string_test) == set(few_shot_tuples[2][1]) or set(
                    class_string_test) == set(few_shot_tuples[3][1]):
                cntt += 1
                column_count += len(class_string_test)

                # Combine examples and answers into the LLM prompt
                response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                        {"role": "system","content": "Your task is to classify the columns of a given table with only one of the following classes that are seperated with comma: sex, category, album, status, origin, format, day, location, notes, duration, nationality, region, club, address, rank, name, position, description, country, state, city, code, symbol, isbn, age, type, gender, team, year, company, result, artist."},
                        {"role": "system","content": "Your instructions are: 1. Look at the columns and the types given to you. 2. Examine the values of the columns. 3. Select a type that best represents the meaning of each column. 4. Answer with the selected type only. the format of the answer should be like this: ['type1', 'type2', 'type3']   Print 'I don't know' if you are not able to find the semantic type."},
                        {"role": "user", "content": few_shot_tuples[0][2]},
                        {"role": "assistant", "content": few_shot_tuples[0][1]},
                        {"role": "user", "content": few_shot_tuples[1][2]},
                        {"role": "assistant", "content": few_shot_tuples[1][1]},
                        {"role": "user", "content": few_shot_tuples[2][2]},
                        {"role": "assistant", "content": few_shot_tuples[2][1]},
                        {"role": "user", "content": few_shot_tuples[3][2]},
                        {"role": "assistant", "content": few_shot_tuples[3][1]},
                        {"role": "user", "content": data_string_test}
                    ]
            )
                output.append(response['choices'][0]["message"]["content"])
            checked_ids.append(test_set["table_id"][i])

            if i % 10 == 0:
                print(f"Processed {i} tables")

    return cntt, column_count


print(RAG_prompting_multiـcolumn(final_dataset, trainset_for_multi_rag_embeds, traiset_for_multi_rag))
