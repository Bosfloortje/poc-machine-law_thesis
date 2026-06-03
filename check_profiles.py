import pandas as pd, json
from pathlib import Path

df = pd.read_csv('analysis/llm_explanations/annotations/results/annotations_long.csv')
print('All unique (wet, approach, model, profile_bsn, profile_name) in annotations:')
cols = df[['wet','approach','model','profile_bsn','profile_name']].drop_duplicates().sort_values(['wet','approach','model'])
print(cols.to_string(index=False))

print()
records = []
with open('analysis/llm_explanations/scripts/evaluation/evaluation_output/eval_results.jsonl', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception:
            continue

print(f'Total eval records: {len(records)}')
eval_df = pd.DataFrame([{
    'law': r.get('law'), 'model': r.get('model'),
    'approach': r.get('approach'), 'profile': r.get('profile')
} for r in records])
print('Eval approaches:', eval_df['approach'].unique())
print('Eval models:', eval_df['model'].unique())
print('Eval laws:', eval_df['law'].unique())
print()
print('Zorgtoeslag haiku in eval:')
sub = eval_df[(eval_df.law=='zorgtoeslag') & (eval_df.model=='haiku')]
print(sub['approach'].value_counts().to_string())
