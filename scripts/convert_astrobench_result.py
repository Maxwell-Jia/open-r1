#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate a JSONL file from LightEval evaluation results.
This script extracts questions and model predictions from a parquet file
and matches them with the original dataset to create a JSONL file.
"""

import argparse
import json
import pandas as pd
from datasets import load_dataset
import re
import ast

def clean_question(text):
    """Clean the question text to match with the original dataset."""
    # Remove the prompt template and extract just the question
    lines = text.strip().split('\n')
    # Skip the first two lines (prompt instructions)
    for i, line in enumerate(lines[2:], 2):
        if line and not line.startswith('A)') and not line.startswith('B)') and not line.startswith('C)') and not line.startswith('D)'):
            return line.strip()
    return ""

def main():
    parser = argparse.ArgumentParser(description='Generate JSONL from LightEval results')
    parser.add_argument('--parquet_file', type=str, required=True, 
                        help='Path to the parquet file with evaluation results')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to the output JSONL file')
    args = parser.parse_args()
    
    # Load the evaluation results
    print(f"Loading evaluation results from {args.parquet_file}")
    results_df = pd.read_parquet(args.parquet_file)
    
    # Load the original dataset
    print("Loading original Astrobench MCQ dataset")
    original_dataset = load_dataset('AstroMLab/Astrobench_MCQ_v1_Public', split='train')
    
    # Create a dictionary for quick lookup of questions
    original_questions = {}
    for item in original_dataset:
        original_questions[item['question']] = item
    
    # Process each example in the results
    output_data = []
    matched_count = 0
    total_count = len(results_df)
    
    for i, row in results_df.iterrows():
        example = row['example']
        
        # Extract the question from the example
        question_text = clean_question(example)
        
        # Try to find a match in the original dataset
        matched_item = None
        
        # Direct match
        if question_text in original_questions:
            matched_item = original_questions[question_text]
        else:
            # Try fuzzy matching if direct match fails
            best_match = None
            best_score = 0
            for orig_q, item in original_questions.items():
                # Simple similarity score based on word overlap
                words1 = set(question_text.lower().split())
                words2 = set(orig_q.lower().split())
                intersection = len(words1.intersection(words2))
                union = len(words1.union(words2))
                if union > 0:
                    score = intersection / union
                    if score > best_score and score > 0.95:  # Threshold for matching
                        best_score = score
                        best_match = item
            
            if best_match:
                matched_item = best_match
        
        if matched_item:
            matched_count += 1
            # Get the model's prediction
            # Handle the case where specifics is a string (convert to dict)
            if isinstance(row['specifics'], str):
                specifics = ast.literal_eval(row['specifics'])
            else:
                specifics = row['specifics']
                
            prediction = specifics['extracted_predictions'][-1]  # Use the last prediction
            
            # Create the output item
            output_item = {
                'question': matched_item['question'],
                'A': matched_item['A'],
                'B': matched_item['B'],
                'C': matched_item['C'],
                'D': matched_item['D'],
                'answer': prediction
            }
            
            output_data.append(output_item)
    
    # Write the output JSONL file
    print(f"Writing {len(output_data)} items to {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Matched {matched_count} out of {total_count} examples ({matched_count/total_count*100:.2f}%)")

if __name__ == "__main__":
    main() 