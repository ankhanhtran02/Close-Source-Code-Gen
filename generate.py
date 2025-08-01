from google import genai
from google.genai import types
import argparse
from datasets import load_dataset
import json
import os
from tqdm import tqdm

SYSTEM_PROMPT = '''You are a helpful coding assistant.''' 

def generate():
  parser = argparse.ArgumentParser(description="Arguments to generate code completion")
  parser.add_argument("--api_key", type=str, required=True, help="API key of Gemini API")
  parser.add_argument("--model", type=str, default="gemini-2.5-pro", help="Gemini model name")
  parser.add_argument("--data", type=str, default="AnhMinhLe/repoexec_comparison_dataset", help="HuggingFace dataset ID")
  parser.add_argument("--split", type=str, default="repoexec_bm25_final", help="HuggingFace dataset split")
  parser.add_argument("--save_path", type=str, default="outputs/repoexec_bm25_final_generated.jsonl")
  parser.add_argument("--continue_last_generation", action="store_true", help="Whether to continue last generation")
  parser.add_argument("--temperature", type=float, default=1.0, help="Controls the randomness of the output. Range:[0.0, 2.0]")
  parser.add_argument("--top_p", type=float, default=1.0, help="The maximum cumulative probability of tokens to consider when sampling")
  parser.add_argument("--top_k", type=int, default=40, help="The maximum number of tokens to consider when sampling")
  parser.add_argument("--candidate_count", type=int, default=1, help="Number of generated responses to return")
  parser.add_argument("--max_output_tokens", type=int, default=2048, help="The maximum number of tokens to include in a response candidate")

  args = parser.parse_args()
  
  ds = load_dataset(args.data, split=args.split)
  
  os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

  if args.continue_last_generation:
    try:
      with open(args.save_path, "r", encoding="utf-8") as f:
        try:
          last_line = list(f)[-1]
          file_last_task_id = json.loads(last_line)["task_id"]
          ds_last_sample_id = -1
          for i, sample in enumerate(ds):
            if sample["id"] == file_last_task_id:
              ds_last_sample_id = i
          if ds_last_sample_id == -1:
            raise Exception("Last task ID not found in dataset.")
          else:
            print(f"Continue generating after task {file_last_task_id}")
            ds = ds.select(range(ds_last_sample_id+1, len(ds)))
        except IndexError:
          print(f"File {args.save_path} is empty.")
    except FileNotFoundError:
      print(f"File {args.save_path} does not exist.")
  else:
    print(f"Start generating from scratch.\nPlease make sure {args.save_path} is empty or not created.")

  client = genai.Client(api_key=args.api_key)
  for sample in tqdm(ds): 
    with open(args.save_path, "a", encoding="utf-8") as f:
      response = client.models.generate_content(
          model=args.model,
          config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            candidate_count=args.candidate_count,
            max_output_tokens=args.max_output_tokens,
          ),
          contents=sample['prompt']
      )

      response_texts = []
      if args.candidate_count == 1:
        response_texts.append(response.text)
      else:
        for candidate in response.candidates:
          response_texts.append(candidate.content.parts[0].text)

      line =  {
          "task_id":sample["id"],
          "prompt":sample["prompt"],
          "response":response_texts
        }
      f.write(json.dumps(line) + "\n")
  print("Done generating.")
  

if __name__ == "__main__":
  generate()