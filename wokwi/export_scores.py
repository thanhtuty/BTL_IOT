"""Export PalmLock's real per-image similarity CSV to Wokwi demo_scores.h.

This copies *measured PC cosine scores* into a replay table. It DOES NOT
convert CompNet/PalmBridge into ESP32 firmware or run images on ESP32.
Usage: python wokwi/export_scores.py --csv test_results/score_per_image.csv
"""
import argparse
import csv
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PEOPLE=[f'person_{i:02d}' for i in range(1,5)]


def convert(source: Path, output: Path):
    samples=[]
    with source.open('r',newline='',encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            if row.get('error'):
                continue
            expected=row.get('expected','')
            if expected not in PEOPLE:
                continue
            label=str(row.get('image',''))
            if not re.fullmatch(r'[a-zA-Z0-9_./\\-]{1,120}',label):
                raise ValueError(f'Unsafe file name: {label!r}')
            vals=[]
            for user in PEOPLE:
                text=row.get(f'{user}_similarity_pct')
                if text is None or text=='':
                    raise ValueError('Missing cosine score '+user+' for '+label)
                try:
                    centipct=int((Decimal(text)*100).to_integral_value())
                except InvalidOperation:
                    raise ValueError('Invalid similarity score for '+label)
                if abs(centipct)>10000:
                    raise ValueError('cosine out of range '+label)
                vals.append(centipct)
            samples.append((label.replace('\\','/'),int(expected[-2:]),vals))
    if not samples:
        raise ValueError('No valid labeled test samples in CSV')
    rows=[]
    for label,person,vals in samples:
        rows.append('  {"'+label+'", '+str(person)+', {'+', '.join(map(str,vals))+'}}')
    text=(
        '#pragma once\n#include <Arduino.h>\n'
        '// REPLAY: Actual PC cosine scores, not on-device AI inference.\n'
        'struct DemoSample { const char *file; uint8_t expected; int16_t score[4]; };\n'
        'static const DemoSample DEMO_SAMPLES[] = {\n'+',\n'.join(rows)+'\n};\n'
        'static const size_t DEMO_COUNT = sizeof(DEMO_SAMPLES)/sizeof(DEMO_SAMPLES[0]);\n'
    )
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(text,encoding='utf-8')
    print(f'Exported {len(samples)} real score traces to {output}')
    top1=sum(vals.index(max(vals))+1==person for _,person,vals in samples)
    print(f'Top-1 on replay data: {top1}/{len(samples)} = {top1/len(samples):.1%}')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--csv',type=Path,default=ROOT/'test_results/score_per_image.csv')
    p.add_argument('--output',type=Path,default=ROOT/'wokwi/demo_scores.h')
    args=p.parse_args()
    convert(args.csv,args.output)


if __name__=='__main__':
    main()
