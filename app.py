import argparse
from typing import Optional, Sequence, Dict, Union, Any, List, Literal
import csv
import jiwer
from collections import defaultdict
from string import Template

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
        'substitute': defaultdict(lambda: 0)
    })

def get_edit_dict(
        reference: str,
        hypothesis: str,
        alignments: List[jiwer.process.AlignmentChunk]
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
    edits = edit_dict_factory()

    ref_i = 0
    hyp_i = 0
    for align in alignments:
        pred_char = hypothesis[hyp_i] if hyp_i < len(hypothesis) else None
        label_char = reference[ref_i] if ref_i < len(reference) else None

        if align.type == 'insert':
            edits[pred_char]['insert']+=1
            hyp_i+=1
        elif align.type == 'delete':
            edits[label_char]['delete']+=1
            ref_i+=1
        elif align.type == 'substitute':
            edits[label_char]['substitute'][pred_char]+=1
            ref_i+=1
            hyp_i+=1
        else: 
            ref_i+=1
            hyp_i+=1
        
    return remove_zero_edits(edits)

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

def remove_zero_edits(d: Dict[str, Dict[str, Any]]):
    """
    Pop any key from edit dict if value is 0.
    """
    for char, edits in d.items():
        for edit, val in list(edits.items()):
            if (val==0) or (edit=='rep' and len(val)==0):
                d[char].pop(edit)
    
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
    for line in lines:
        ref, hyp = line[:2]
        cer = jiwer.process_characters(ref, hyp)
        row=get_edit_html(ref, hyp, metric=cer)
        cer_data.append(row)

    # visualize WER for each record
    wer_data = []
    for line in lines:
        ref, hyp = line[:2]
        wer = jiwer.process_words(ref, hyp)
        row=get_edit_html(ref, hyp, metric=wer)
        wer_data.append(row)

    # add into html header and save
    full_html = header_template.substitute(
        cer_content="\n".join(cer_data),
        wer_content="\n".join(wer_data),
    )
    html_out = args.html or args.input.replace('.csv', '.html')
    with open(html_out, 'w') as f:
        f.write(full_html)

    return 0

if __name__ == '__main__':
    main()