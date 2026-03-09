import base64
import os
import re
import pandas as pd
import numpy as np
import ast
from pandarallel import pandarallel
from tqdm import tqdm
import argparse
from openai import OpenAI

# Initialize parallel processing
pandarallel.initialize(progress_bar=True, nb_workers=1)

# -------------------------- 1. Image Encoding --------------------------
def encode_image(image_path):
    """Encode image file to Base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# -------------------------- 2. Model Inference --------------------------
def get_response(image_path, api_key, model_name="qwen-vl-plus-latest", max_tokens=200, temperature=1.2, n=1):
    """Basic geo-localization prediction (No RAG)."""
    base64_image = encode_image(image_path)
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text":  f"""your prompts"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "low"
                            }
                        }
                    ]
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            n=n
        )

        ans = []
        for choice in completion.choices:
            ans.append(choice.message.content)
        return ans

    except Exception as e:
        print(f"Model call error: {e}")
        return ['{"latitude": 0.0,"longitude": 0.0}'] * n

def get_response_rag(image_path, api_key, candidates_gps, reverse_gps, model_name="qwen-vl-plus-latest", max_tokens=200, temperature=1.2, n=1):
    """RAG-enhanced geo-localization prediction."""
    base64_image = encode_image(image_path)
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # Construct prompt with reference coordinates
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text":  f"""your prompts"""
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "low"
                    }
                }
            ]
        }
    ]

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            n=n
        )
        
        if not hasattr(completion, 'choices') or len(completion.choices) == 0:
            raise Exception(f"Invalid response choices: {completion}")
        
        ans = []
        for choice in completion.choices:
            if not hasattr(choice, 'message') or not hasattr(choice.message, 'content'):
                raise Exception(f"Invalid choice structure: {choice}")
            ans.append(choice.message.content)
        return ans

    except Exception as e:
        raise Exception(f"RAG model call failed: {str(e)}")

# -------------------------- 3. Row Processing --------------------------
def process_row(row, api_key, model_name, root_path, image_path, n=4):
    """Process single row (No RAG)."""
    full_image_path = os.path.join(root_path, image_path, row["IMG_ID"])
    try:
        response = get_response(full_image_path, api_key, model_name, n=n)
    except Exception as e:
        response = "None"
        print(f"Error processing image {row['IMG_ID']}: {e}")
    row['response'] = response
    return row

def process_row_rag(row, api_key, model_name, root_path, image_path, rag_sample_num, n=4):
    """Process single row (RAG-enhanced)."""
    full_image_path = os.path.join(root_path, image_path, row["IMG_ID"])
    try:
        # Extract candidate and reverse GPS lists
        candidates_gps = [row[f'candidate_{i}_gps'] for i in range(rag_sample_num)]
        reverse_gps = [row[f'reverse_{i}_gps'] for i in range(rag_sample_num)]
        
        response = get_response_rag(full_image_path, api_key, str(candidates_gps), str(reverse_gps), model_name, n=n)
    except Exception as e:
        response = "None"
        print(f"RAG error processing image {row['IMG_ID']}: {e}")
    row['rag_response'] = response
    return row

# -------------------------- 4. Helpers --------------------------
def check_conditions(coord_str):
    """Check if coordinates are invalid (0.0 or empty)."""
    if coord_str.startswith('[]') or coord_str.startswith('None'):
        return True
    try:
        coordinates = ast.literal_eval(coord_str)
        return float(coordinates[0]) == 0.0
    except:
        return True

# -------------------------- 5. Main Execution Logic --------------------------
def run(args):
    api_key = args.api_key
    model_name = args.model_name
    root_path = args.root_path
    text_path = args.text_path
    image_path = args.image_path
    result_path = args.result_path
    rag_path = args.rag_path
    process = args.process
    rag_sample_num = args.rag_sample_num
    searching_file_name = args.searching_file_name
    n = args.n

    # --- 5.1 Basic Prediction (predict) ---
    if process == 'predict':
        result_full_path = os.path.join(root_path, result_path)
        if os.path.exists(result_full_path):
            df = pd.read_csv(result_full_path)
            # Filter rows needing retry (None/Empty)
            df_rerun = df[df['response'].isna() | (df['response'] == 'None') | (df['response'] == '')]
            print(f'Rows to retry (predict): {df_rerun.shape[0]}')
            
            if not df_rerun.empty:
                df_rerun = df_rerun.parallel_apply(
                    lambda row: process_row(row, api_key, model_name, root_path, image_path, n),
                    axis=1
                )
                df.update(df_rerun)
                df.to_csv(result_full_path, index=False)
        else:
            df = pd.read_csv(text_path)
            df['response'] = pd.Series(dtype=str)
            df = df.parallel_apply(
                lambda row: process_row(row, api_key, model_name, root_path, image_path, n),
                axis=1
            )
            df.to_csv(result_full_path, index=False)
        print(f"Basic predictions saved to: {result_full_path}")

    # --- 5.2 Coordinate Extraction (extract) ---
    elif process == 'extract':
        result_full_path = os.path.join(root_path, result_path)
        df = pd.read_csv(result_full_path)
        pattern = r'[-+]?\d+\.\d+'
        # Extract floats from response text
        df['coordinates'] = df['response'].apply(lambda x: re.findall(pattern, x) if x not in ['None', ''] else [])
        df.to_csv(result_full_path, index=False)
        print(f"Basic coordinates extracted to: {result_full_path}")

    # --- 5.3 RAG Prediction (rag) ---
    elif process == 'rag':
        database_df = pd.read_csv(os.path.join(os.path.dirname(text_path), ""))
        rag_result_full_path = os.path.join(root_path, f"{rag_sample_num}_{rag_path}")

        if not os.path.exists(rag_result_full_path):
            # Initial run: Load data and indices
            df = pd.read_csv(text_path)
            I = np.load(searching_file_name)
            reverse_I = np.load(f"{os.path.splitext(searching_file_name)[0]}_reverse.npy")

            # Populate RAG reference columns
            for i in tqdm(range(df.shape[0]), desc="Generating RAG references"):
                # Similar candidates
                candidate_idx_lis = I[i][:rag_sample_num]
                candidate_gps = database_df.loc[candidate_idx_lis, ['LAT', 'LON']].values
                for idx, (latitude, longitude) in enumerate(candidate_gps):
                    df.loc[i, f'candidate_{idx}_gps'] = f'[{latitude}, {longitude}]'
                
                # Dissimilar candidates
                reverse_idx_lis = reverse_I[i][:rag_sample_num]
                reverse_gps = database_df.loc[reverse_idx_lis, ['LAT', 'LON']].values
                for idx, (latitude, longitude) in enumerate(reverse_gps):
                    df.loc[i, f'reverse_{idx}_gps'] = f'[{latitude}, {longitude}]'

            df['rag_response'] = pd.Series(dtype=str)
            df.to_csv(rag_result_full_path, index=False)
            
            # Run inference
            df = df.parallel_apply(
                lambda row: process_row_rag(row, api_key, model_name, root_path, image_path, rag_sample_num, n),
                axis=1
            )
            df.to_csv(rag_result_full_path, index=False)
        else:
            # Resume run: Filter incomplete rows
            df = pd.read_csv(rag_result_full_path)
            
            if 'rag_response' not in df.columns:
                df['rag_response'] = pd.Series(dtype=str)
            
            df_rerun = df[df['rag_response'].isna() | (df['rag_response'] == 'None') | (df['rag_response'] == '')]
            print(f'Rows to retry (RAG): {df_rerun.shape[0]}')
            
            if not df_rerun.empty:
                df_rerun = df_rerun.parallel_apply(
                    lambda row: process_row_rag(row, api_key, model_name, root_path, image_path, rag_sample_num, n),
                    axis=1
                )
                df.update(df_rerun)
                df.to_csv(rag_result_full_path, index=False)
        
        print(f"RAG predictions saved to: {rag_result_full_path}")

    # --- 5.4 RAG Coordinate Extraction (rag_extract) ---
    elif process == 'rag_extract':
        # Hardcoded path override for specific environment (adjust as needed)
        rag_result_full_path = """"""
        df = pd.read_csv(rag_result_full_path).fillna("None")
        pattern = r'[-+]?\d+\.\d+'
        df['rag_coordinates'] = df['rag_response'].apply(lambda x: re.findall(pattern, x) if x not in ['None', ''] else [])
        df.to_csv(rag_result_full_path, index=False)
        print(f"RAG coordinates extracted to: {rag_result_full_path}")

# -------------------------- 6. Entry Point --------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Qwen-VL Geo-localization")
    
    # Model Args
    parser.add_argument('--api_key', type=str, default="", 
                        help="Aliyun DashScope API Key")
    parser.add_argument('--model_name', type=str, default="qwen-vl-plus", 
                        help="Model name")
    parser.add_argument('--n', type=int, default=1, choices=[1,2,3,4],
                        help="Number of generations per image")
    
    # Path Args
    parser.add_argument('--root_path', type=str, default="", 
                        help="Root path for images")
    parser.add_argument('--text_path', type=str, 
                        default="",
                        help="Path to test set CSV")
    parser.add_argument('--image_path', type=str, default="", 
                        help="Subfolder for images")
    parser.add_argument('--result_path', type=str, default="", 
                        help="Output filename for basic prediction")
    parser.add_argument('--rag_path', type=str, default="", 
                        help="Output filename for RAG prediction")
    parser.add_argument('--searching_file_name', type=str, 
                        default="", 
                        help="Path to retrieval index")
    
    # Control Args
    args = parser.parse_args()
    

    run(args)

# Note: Full documentation will be updated upon official acceptance.