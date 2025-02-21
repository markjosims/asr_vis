import argparse
from typing import Optional, Sequence, Dict, Union, Any, List, Literal
import csv
import jiwer
from collections import defaultdict
from string import Template
import json
from itertools import zip_longest

WORD_METRICS = ['wer', 'mer', 'wil', 'wip']

# ----------------- #
# edit dict helpers #
# ----------------- #

def edit_dict_factory():
    """
    Build a `defaultdict` such that when adding a new key for a word,
    returns a dict with keys for `insert`, `delete` and `substitute` edits.
    For `insert` and `delete`, default value is 0.
    For `substitute`, default value is another `defaultdict` that returns 0.
    """
    return defaultdict(lambda: {
        'insert': 0,
        'delete': 0,
        'reference_ct': 0,
        'hypothesis_ct': 0,
        'substitute': defaultdict(lambda: {'ct': 0})
    })

def get_edit_dict(
        reference: str,
        hypothesis: str,
        alignments: List[jiwer.process.AlignmentChunk],
        edit_dict = None,
    ) -> Dict[
            str,
            Dict[
                str, Union[int, Dict[str, int]]
            ]
    ]:
    """
    Given a reference str, hypothesis str and jiwer character alignments,
    create a dictionary with counts of each type of edit for each unique character.
    """
    if edit_dict is None:
        edit_dict = edit_dict_factory()

    for align in alignments:
        # string of chars for CER or list of words for WER
        hyp_chunk = hypothesis[align.hyp_start_idx:align.hyp_end_idx]
        ref_chunk = reference[align.ref_start_idx:align.ref_end_idx]
        for hyp_substr, ref_substr in zip_longest(hyp_chunk, ref_chunk, fillvalue=None):
            if hyp_substr:
                edit_dict[hyp_substr]['hypothesis_ct']+=1
            if ref_substr:
                edit_dict[ref_substr]['reference_ct']+=1
            if align.type == 'insert':
                edit_dict[hyp_substr]['insert']+=1
            elif align.type == 'delete':
                edit_dict[ref_substr]['delete']+=1
            elif align.type == 'substitute':
                edit_dict[ref_substr]['substitute'][hyp_substr]['ct']+=1
    return edit_dict

def merge_edit_dicts(
        main_dict: Dict[str, Dict[str, Any]],
        incoming: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
    """
    Merge edit counts for two edit dicts.
    """
    for char, edits in incoming.items():
        for edit, val in edits.items():
            if type(val) is int:
                main_dict[char][edit]+=val
            else:
                for tgt_char, ct in val.items():
                    main_dict[char][edit][tgt_char]+=ct
    
    return main_dict

def add_rate_keys(d: Dict[str, Dict[str, Any]]):
    """
    For each edit add key `{edit}_rate`.
    - insert_rate: percent of all instances of X in hypothesis that are insertions
    - delete_rate: percent of all instances of X in reference that were deleted in hypothesis
    - substitute_rate: for substitution X>Y, percent of instances of X in reference
      substituted to Y in hypothesis
    """
    for substr, edits in d.items():
        hyp_ct = edits['hypothesis_ct']
        ref_ct = edits['reference_ct']
        insert_ct = edits['insert']
        d[substr]['insert_rate']=insert_ct/hyp_ct if insert_ct else 0
        delete_ct = edits['delete']
        d[substr]['delete_rate']=delete_ct/ref_ct if delete_ct else 0
        for hyp_substr, sub_edits in edits['substitute'].items():
            sub_ct = sub_edits['ct']
            d[substr]['substitute'][hyp_substr]['rate']=sub_ct/ref_ct if sub_ct else 0
    return d

def remove_zero_edits(d: Dict[str, Dict[str, Any]]):
    """
    Pop any key from edit dict if value is 0.
    """
    for substr, edits in d.items():
        for edit, val in list(edits.items()):
            if (val==0) or (edit=='substitute' and len(val)==0):
                d[substr].pop(edit)
    
    return d

# ------------ #
# html helpers #
# ------------ #

def get_edit_html(
        reference: str,
        hypothesis: str,
        metric: Union[jiwer.process.WordOutput, jiwer.process.CharacterOutput]
    ):
    metric_type = 'wer' if type(metric) is jiwer.process.WordOutput else 'cer'
    row = Template(f"""
<div class ="record">
    <div class="reference">Reference: {reference}</div>
    <div class="hypothesis">Hypothesis: {hypothesis}</div>
    $metrics
    <table>
        <tr>
            <td>Reference:</td>
            $ref_data
        </tr>
        <tr>
            <td>Hypothesis:</td>
            $hyp_data
        </tr>
    </table>
</div>
    """)
    # get overall metrics
    metric_data = []
    if metric_type=='wer':
        # metric list
        for word_metric in WORD_METRICS:
            metric_data.append((word_metric,getattr(metric, word_metric)))
        # set reference and hypothesis to lists of words rather than strs
        reference = reference.split()
        hypothesis = hypothesis.split()
    else:
        metric_data.append(('cer',metric.cer))
    metric_html = [f"""<div class="metric">{name}: {val}</div>""" for name, val in metric_data]

    # generate edit table
    ref_data = []
    hyp_data = []
    for alignment in metric.alignments[0]:
        ref_chunk = reference[alignment.ref_start_idx:alignment.ref_end_idx]
        hyp_chunk = hypothesis[alignment.hyp_start_idx:alignment.hyp_end_idx]
        if metric_type == 'cer':
            # represent spaces as underscores for visibility
            ref_chunk = ref_chunk.replace(' ', '_')
            hyp_chunk = hyp_chunk.replace(' ', '_')
        if metric_type == 'wer':
            # don't print a list
            ref_chunk = ' '.join(ref_chunk)
            hyp_chunk = ' '.join(hyp_chunk)
        ref_data.append(f"""<td class="{alignment.type}">{ref_chunk}</td>""")
        hyp_data.append(f"""<td class="{alignment.type}">{hyp_chunk}</td>""")
    row = row.substitute(
        ref_data="\n".join(ref_data),
        hyp_data="\n".join(hyp_data),
        metrics="\n".join(metric_html),
    )
    return row

def make_insert_delete_table(edit_dict):
    table = Template("""
<table>
    <tr>
        <td>String</td>
        <td>Insert</td>
        <td>%</td>
        <td>Delete</td>
        <td>%</td>
        <td>Num Ref</td>
        <td>Num Hyp</td>
    </tr>
    $table_data
</table>
""")
    row = Template("""
<tr>
    <td>$string</td>
    <td>$insert</td>
    <td>$insert_rate</td>
    <td>$delete</td>
    <td>$delete_rate</td>
    <td>$reference_ct</td>
    <td>$hypothesis_ct</td>
</tr>
""")
    rows = [row.safe_substitute({'string': k, **v}) for k,v in edit_dict.items()]
    return table.substitute(table_data="\n".join(rows))

# ----- #
# script #
# ----- #

def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser('Generate summary of ASR errors.')
    parser.add_argument('--input', '-i')
    parser.add_argument('--html', '-H')
    parser.add_argument('--json', '-j')
    return parser

def main(argv: Optional[Sequence[str]]=None) -> int:
    parser = init_parser()
    args = parser.parse_args(argv)

    # read in html header
    with open('static/header.html') as html_file:
        header_template = Template(html_file.read())

    # read in csv data
    with open(args.input) as csvfile:
        reader = csv.reader(csvfile)
        lines = [row for row in reader]
    
    # validate csv header
    header = lines.pop(0)
    if (header!=['reference','hypothesis']) and (header!=['reference','hypothesis','audio']):
        raise ValueError("Input .csv file must have header `reference,hypothesis` or `reference,hypothesis,audio`")
    
    # visualize CER for each record
    cer_data = []
    char_edits = edit_dict_factory()
    for line in lines:
        ref, hyp = line[:2]
        cer = jiwer.process_characters(ref, hyp)
        row=get_edit_html(ref, hyp, metric=cer)
        cer_data.append(row)
        char_edits = get_edit_dict(ref, hyp, cer.alignments[0], char_edits)
    char_edits = add_rate_keys(char_edits)
    char_edit_table = make_insert_delete_table(char_edits)

    # visualize WER for each record
    wer_data = []
    word_edits = edit_dict_factory()
    for line in lines:
        ref, hyp = line[:2]
        wer = jiwer.process_words(ref, hyp)
        row=get_edit_html(ref, hyp, metric=wer)
        wer_data.append(row)
        word_edits = get_edit_dict(ref.split(), hyp.split(), wer.alignments[0], word_edits)
    word_edits = add_rate_keys(word_edits)
    word_edit_table = make_insert_delete_table(word_edits)

    # add into html header and save
    full_html = header_template.substitute(
        cer_content="\n".join(cer_data),
        wer_content="\n".join(wer_data),
        char_edit_content=char_edit_table,
        word_edit_content=word_edit_table,
    )
    html_out = args.html or args.input.replace('.csv', '.html')
    with open(html_out, 'w') as f:
        f.write(full_html)

    char_edits = remove_zero_edits(char_edits)
    word_edits = remove_zero_edits(word_edits)
    full_json = {'char_edits': char_edits, 'word_edits': word_edits}
    json_out = args.json or args.input.replace('.csv', '.json')
    with open(json_out, 'w') as f:
        json.dump(full_json, f, indent=2, ensure_ascii=False)
    return 0

if __name__ == '__main__':
    main()