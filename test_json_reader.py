import json

file_path = r"C:\Users\Adweraveth\Desktop\Denemeler BA\model_results.jsonl"

with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line)
        print(data)