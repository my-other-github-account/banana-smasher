"""Compare calibration token content to an immutable evaluation ledger.

Different row IDs alone never establish separation. This gate rejects complete
window containment and reports every pair's longest exact token span, without
inventing a numeric overlap tolerance or rewriting either corpus. Admission of
partial spans additionally requires inspection and independent document ancestry.
"""
from difflib import SequenceMatcher


def validate_separation(calibration, evaluation):
    if not calibration or not evaluation:
        raise ValueError('both calibration and evaluation rows are required')
    train_ids = [str(r['item_id']) for r in calibration]
    eval_ids = [str(r['item_id']) for r in evaluation]
    overlap = sorted(set(train_ids) & set(eval_ids))
    if overlap or len(train_ids) != len(set(train_ids)):
        raise ValueError(f'calibration ID overlap: {overlap}')
    matches = []
    contained = []
    for e in evaluation:
        et = e['token_ids']
        if not et:
            raise ValueError('empty evaluation content')
        matcher = SequenceMatcher(None, [], et, autojunk=False)
        for c in calibration:
            ct = c['token_ids']
            if not ct:
                raise ValueError('empty calibration content')
            matcher.set_seq1(ct)
            m = matcher.find_longest_match(0, len(ct), 0, len(et))
            row = {'calibration_id': c['item_id'], 'evaluation_id': e['item_id'],
                   'calibration_offset': m.a, 'evaluation_offset': m.b,
                   'tokens': m.size, 'token_ids': ct[m.a:m.a + m.size]}
            matches.append(row)
            if m.size == min(len(ct), len(et)):
                contained.append(row)
    if contained:
        raise ValueError(f'complete window content overlap: {contained}')
    matches.sort(key=lambda r: r['tokens'], reverse=True)
    return {'id_overlap': overlap, 'full_window_overlap': contained,
            'pairs_compared': len(matches),
            'maximum_contiguous_match_tokens': matches[0]['tokens'],
            'longest_matches': matches, 'arbitrary_overlap_threshold': None,
            'document_ancestry_required': True}
