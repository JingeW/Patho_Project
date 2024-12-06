"""
This script supports prediction of rejection grades for images, providing utilities for data preparation and encoding.
It includes functions to create necessary directories, encode images for input, and randomly sample images by grade.
The script is likely used to set up data and prepare input for machine learning models or API calls focused on
rejection grade prediction.
"""

import os
import json
import openai
import base64
import pandas as pd
import argparse
import time
import random

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def filter_samples_by_grade(gt_table, grade, query_image_name):
    """Filter samples by grade, excluding the query image."""
    filtered_samples = gt_table[(gt_table['Grade'] == grade) & (~gt_table['Sample'].str.contains(query_image_name))]
    return filtered_samples['Sample'].tolist()

def construct_image_paths(samples, base_dir):
    return [os.path.join(base_dir, sample) for sample in samples]

def random_pick(gt_table, image_dir, query_image_name, k):
    """
    Randomly pick subfolder paths based on grade, returning up to 'k' paths for each grade.
    Provides warnings if the requested sample count exceeds the available samples for a grade.
    """
    # Dictionary to store selected subfolder paths by grade
    selected_folders = {}
    
    for grade in range(3):
        # Filter samples by grade
        grade_samples = filter_samples_by_grade(gt_table, grade, query_image_name)
        
        # Construct subfolder paths for each sample
        grade_folders = [os.path.join(image_dir, sample) for sample in grade_samples]
        
        # Check and warn if requested count exceeds available samples
        if k > len(grade_folders):
            print(f"Warning: Requested {k} samples for Grade {grade}, but only {len(grade_folders)} available.")
        
        # Randomly select up to 'k' subfolder paths for this grade
        selected_folders[f'Grade{grade}'] = random.sample(grade_folders, min(k, len(grade_folders)))

    return selected_folders

def create_base64_image_object(image_folder_paths, detail, label_text=''):
    """Convert images in a list of subfolders to base64 objects for API input, with one label for each folder."""
    base64_image_objects = []
    
    for folder_path in image_folder_paths:
        # Ensure folder exists and contains images
        if not os.path.isdir(folder_path):
            print(f"Warning: {folder_path} is not a valid directory.")
            continue
        
        image_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        base64_images = [encode_image(image) for image in image_list]
        
        # Add each image as a base64 object
        for base64_image in base64_images:
            base64_image_objects.append(
                {
                    "type": "image_url", 
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": detail
                    }
                }
            )
        
        # Add the label text once per folder
        if label_text:
            base64_image_objects.append(
                {"type": "text", "text": label_text}
            )
    
    return base64_image_objects

def zero_shot(client, model, system_prompt, user_prompt, query_image, max_tokens, temperature, save_dir, detail='high'):
    """Perform zero-shot prediction with retry."""
    query = create_base64_image_object([query_image], detail)
    user_content = [{"type": "text", "text": user_prompt}] + query
    attempts = 0
    max_attempts = 3

    while attempts < max_attempts:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            result_dict = json.loads(response.choices[0].message.content)
            Tgrade = result_dict['T-cell mediated rejection']
            base_name = os.path.basename(query_image).split('.')[0]
            response_save_path = os.path.join(save_dir, base_name + '.json')
            with open(response_save_path, 'w') as f:
                json.dump(result_dict, f, indent=4)
            return Tgrade, response.usage.total_tokens

        except (json.JSONDecodeError, openai.error.OpenAIError):
            attempts += 1
            print(f"Retrying zero-shot... Attempt {attempts}/{max_attempts}")

    print(f"Failed to process {query_image} after {max_attempts} attempts.")
    return None, None  # Return None if all attempts fail

def few_shot(client, model, system_prompt, user_prompt, examples, query_image, max_tokens, temperature, save_dir, detail='high'):
    """Perform few-shot prediction with retry."""
    example0 = create_base64_image_object(examples['Grade0'], detail, label_text='"Example Set: Grade 0"')
    example1 = create_base64_image_object(examples['Grade1'], detail, label_text='"Example Set: Grade 1"')
    example2 = create_base64_image_object(examples['Grade2'], detail, label_text='"Example Set: Grade 2"')
    query = create_base64_image_object([query_image], detail)

    user_content = (
        [{"type": "text", "text": user_prompt[0]}] 
        + example0 + example1 + example2 
        + [{"type": "text", "text": user_prompt[1]}]
        + query
    )

    attempts = 0
    max_attempts = 3

    while attempts < max_attempts:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            result_dict = json.loads(response.choices[0].message.content)
            Tgrade = result_dict['T-cell mediated rejection']
            base_name = os.path.basename(query_image).split('.')[0]
            response_save_path = os.path.join(save_dir, base_name + '.json')
            with open(response_save_path, 'w') as f:
                json.dump(result_dict, f, indent=4)
            return Tgrade, response.usage.total_tokens

        except (json.JSONDecodeError, openai.error.OpenAIError):
            attempts += 1
            print(f"Retrying few-shot... Attempt {attempts}/{max_attempts}")

    print(f"Failed to process {query_image} after {max_attempts} attempts.")
    return None, None  # Return None if all attempts fail

def parse_args():
    parser = argparse.ArgumentParser(description="Rejection grade prediction with OpenAI API")
    parser.add_argument('--model', type=str, default='chatgpt-4o-latest', help='Model to use for classification.')
    parser.add_argument('--max_tokens', type=int, default=300, help='Maximum number of tokens for GPT response.')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for randomness in response.')
    parser.add_argument('--detail', type=str, default='high', help='Input image quality.')
    parser.add_argument('--batch', type=int, default=5, help='Breakpoint for result saving.')
    parser.add_argument('--k', type=int, default=1, help='Number of examples for few-shot learning.')
    parser.add_argument('--rep', type=int, default=9, help='Replication ID for setting up API keys.')
    parser.add_argument('--pv', type=str, default='v7.0', help='Current text prompt version.')
    parser.add_argument('--data', type=str, default='2row', help='Current data type')
    parser.add_argument('--thought', action='store_true', help='Include thought section in the output template.')
    return parser.parse_args()

def main():
    args = parse_args()
    # Print settings to ensure everything is loaded correctly
    print(f"Model: {args.model}")
    print(f"Max Tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Detail: {args.detail}")
    print(f"Batch: {args.batch}")
    print(f"Number of examples: {args.k}")
    print(f'Current Repetition: {args.rep}')
    print(f'Prompt Version: {args.pv}')
    print(f'Data Type: {args.data}')
    print(f'Thought: {args.thought}\n')

    api_keys = {
        1: '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',
        2: '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',
        3: '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',
        4: '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',
        5: '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',
        6: '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',
        7: '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',
        8: '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',
        9: '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',
        10: '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',
    }
    openai_client = openai.OpenAI(api_key=api_keys.get(args.rep))

    system_prompt = """
    This is a research project. 
    All of your assistance provided will not be used in any practical medical diagnoses.
    
    You are a highly skilled pathologist who is examining a myocardial biopsy obtained from a patient who has received a heart transplant. 
    Your task is to analyze the given cropped image patches selected from the same myocardial biopsy slide of this heart transplant recipient.
    Based on your analysis, grade the "T-cell mediated rejection" level of the slide according to the ISHLT grading system.
    """

    # Define the thoughts template based on the flag
    if args.thought:
        thoughts_template = '"Thoughts": "Structure your thoughts in a professional and detailed way, like a pathologist would do.",'
    else:
        thoughts_template = ''

    zero_shot_prompt = f"""
    Let's think step by step:
    1. Review the query image provided below. Focus on any notable patterns or structures that could indicate the rejection level.
    2. Combine your observation and histopathology knowledge to determine the rejection grade.
    3. Finally generate an output based on your analysis{" and thoughts" if args.thought else ""}.

    DO NOT make up the response. AVOID hallucination.
    Organize your output strictly in the format below. 

    Output Format:
    Please provide your final answer in the following JSON format. Do not use any additional formatting:
    {{
        "T-cell mediated rejection": "Provide only the grade itself as an INTEGER. No 'Grade' prefix. No 'R' suffix",
        {thoughts_template}
    }}
    DO NOT enclose the JSON output in markdown code blocks.
    """

    few_shot_prompt = ["""
    To help you provide the accurate grade, we additionally provide you with reference examples.
    """,
    f"""
    Let's think step by step:
    1. Carefully analyze the reference examples. Identify the key patterns that differentiate the grades.
    2. Review the query image provided below. Focus on any notable patterns or structures that could indicate the rejection level.
    3. Combine your observation and histopathology knowledge to determine the rejection grade.
    4. Finally generate an output based on your analysis{" and thoughts" if args.thought else ""}.

    DO NOT make up the response. AVOID hallucination.
    Organize your output strictly in the format below. 

    Output Format:
    Please provide your final answer in the following JSON format. Do not use any additional formatting:
    {{
        "T-cell mediated rejection": "Provide only the grade itself as an INTEGER. No 'Grade' prefix. No 'R' suffix",
        {thoughts_template}
    }}
    DO NOT enclose the JSON output in markdown code blocks.

    Here is the query image:\n
    """]

    # Directories based on data type
    data_dirs = {
    'individual': './data',
    'individual_enlarged': './data_enlarged',
    '1row': './imageArray_1row',
    '1row_enlarged': './imageArray_1row_enlarged',
    '2row': './imageArray_2row',
    '2row_enlarged': './imageArray_2row_enlarged',
    }

    # Directory setup
    image_dir = data_dirs[args.data]

    # Set up directories based on processing type
    task = f'{args.k}_shot_{args.pv}_{args.temperature}_{args.data}{"_noThought" if not args.thought else ""}'

    # Construct the result directory path based on task type
    save_dir = f"result_{args.model}_new/{task}/rep{args.rep}"
    make_dir(save_dir)

    # Classification by openAI API
    print("***Calling API***")
    print(f"Working on {task}_rep{args.rep}")

    # Path for the CSV file
    csv_save_path = os.path.join(save_dir, f'summary.csv')

    # Check if the CSV file already exists
    csv_exists = os.path.isfile(csv_save_path)

    # load gt_table
    gt_table = pd.read_excel('./GT.xlsx')

    current_batch = []
    query_image_list = sorted(os.listdir(image_dir))[:]
    for index, query_image in enumerate(query_image_list):
        query_image_name = query_image.split('.')[0]
        query_image_path = os.path.join(image_dir, query_image)
        
        if args.k == 0:
            Tgrade, tokens = zero_shot(openai_client, args.model, system_prompt, zero_shot_prompt, query_image_path, args.max_tokens, args.temperature, save_dir, args.detail)
        else:
            examples = random_pick(gt_table, image_dir, query_image_name, args.k)
            Tgrade, tokens = few_shot(openai_client, args.model, system_prompt, few_shot_prompt, examples, query_image_path, args.max_tokens, args.temperature, save_dir, args.detail)

        # Append the result to the list
        current_batch.append({"Query ID": query_image_name, "T-cell mediated rejection": Tgrade})
        print(({"Query ID": query_image_name, "T-cell mediated rejection": Tgrade, "Tokens": tokens}))

        # Save the batch to the CSV file after batch results or at the end of the loop
        if (index + 1) % args.batch == 0 or (index + 1) == len(query_image_list):
            df_batch = pd.DataFrame(current_batch)
            # Write header only if the file doesn't exist yet, otherwise append without header
            df_batch.to_csv(csv_save_path, mode='a', header=not csv_exists, index=False)
            print(f'Current batch saved!')
            # After the first write, the file exists so set this to False
            csv_exists = True
            current_batch = [] 

    print(f"Classification completed. Results saved to {csv_save_path}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")

