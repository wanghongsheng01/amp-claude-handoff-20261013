"""Thin R/Z/C slot wrapper; original frozen feature/JS formulas unchanged."""
import importlib.util
from collections import Counter
from fractions import Fraction
from pathlib import Path

P = Path(__file__).resolve().parent.parent/'implementation/frozen_metrics_v1.py'
spec=importlib.util.spec_from_file_location('frozen_metrics', P)
frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)

def edit_distance(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1): cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=cur
    return prev[-1]

def score_generation(slots,train,dev,check=lambda:None):
    summaries={};nearest={}
    trainseq=[r['sequence'] for r in train]
    trainset=set(trainseq);trainpair={(r['sequence'],r['terminal_state']) for r in train}
    for arm in ('R','Z','C'):
        summaries[arm]={}
        for terminal in ('FREE','AMD'):
            check()
            rr=[r for r in slots if r['arm']==arm and r['terminal']==terminal]
            assert len(rr)==16 and sorted(r['slot'] for r in rr)==list(range(16))
            completed=[r for r in rr if r['sequence'] is not None and r['status'] in ('eos','cap_nonEOS','empty_sequence','historical_ok')]
            seq=[r['sequence'] for r in completed]
            valid=[s for s in seq if 1<=len(s)<=48 and set(s)<=frozen.AA_SET]
            ref=[r['sequence'] for r in dev if r['terminal_state']==terminal]
            summary=frozen.sequence_summary(valid)
            comp=frozen.compare_summaries(summary,frozen.sequence_summary(ref))
            for r in completed:
                s=r['sequence'];check()
                if s not in nearest:
                    distances=[edit_distance(s,t)/max(len(s),len(t)) for t in trainseq]
                    j=min(range(len(distances)),key=distances.__getitem__)
                    nearest[s]={'normalized_Levenshtein':distances[j],'train_row':j,'train_sequence':trainseq[j]}
                r['nearest_train_edit']=nearest[s]
                r['exact_copy_train']=s in trainset
                r['exact_copy_train_same_terminal']=(s,terminal) in trainpair
            lengthmean=sum(map(len,seq))/len(seq) if seq else None
            duplicate=len(seq)-len(set(seq))
            summaries[arm][terminal]={'requests':16,'completed_n':len(seq),'valid_nonempty_n':len(valid),
                'main_complete':len(completed)==16 and len(valid)==16,'status_counts':dict(Counter(r['status'] for r in rr)),
                'empty_n':sum(s=='' for s in seq),'length_mean_including_completed_empty':lengthmean,
                'length_mean_difference_absolute':abs(lengthmean-sum(map(len,ref))/len(ref)) if lengthmean is not None else None,
                'legal5_48_n':sum(5<=len(s)<=48 and set(s)<=frozen.AA_SET for s in seq),
                'legal5_48_rate':sum(5<=len(s)<=48 and set(s)<=frozen.AA_SET for s in seq)/16,
                'copy_train_n':sum(s in trainset for s in seq),'copy_train_rate':sum(s in trainset for s in seq)/16,
                'copy_train_terminal_n':sum((s,terminal) in trainpair for s in seq),
                'duplicates_n':duplicate,'duplicates_rate':duplicate/16,
                'cap_nonEOS_n':sum(r['status']=='cap_nonEOS' for r in rr),
                'cap_nonEOS_rate':sum(r['status']=='cap_nonEOS' for r in rr)/16,
                'conditional_valid_summary':summary,'conditional_valid_vs_dev':comp}
    assert len(slots)==96
    candidate_ok=all(summaries['Z'][t]['main_complete'] for t in ('FREE','AMD'))
    controls_ok=all(summaries[a][t]['main_complete'] for a in ('R','C') for t in ('FREE','AMD'))
    comparisons={}
    for control in ('R','C'):
        comparisons[control]=[]
        for terminal in ('FREE','AMD'):
            z=summaries['Z'][terminal];b=summaries[control][terminal];checks={}
            x=z['conditional_valid_vs_dev']['mmd2_linear_biased_fixed_features']
            y=b['conditional_valid_vs_dev']['mmd2_linear_biased_fixed_features']
            checks['MMD_strict_and_at_least_10percent']=x is not None and y is not None and x<y and x<=.90*y
            x=z['conditional_valid_vs_dev']['dipeptide_jensen_shannon_bits'];y=b['conditional_valid_vs_dev']['dipeptide_jensen_shannon_bits']
            checks['JS_strict']=x is not None and y is not None and x<y
            for metric in ('length_mean_difference_absolute','copy_train_n','duplicates_rate'):
                x,y=z[metric],b[metric];checks[metric]=x is not None and y is not None and x<=y
            checks['legal5_48_rate']=z['legal5_48_rate']>=b['legal5_48_rate']
            comparisons[control].append({'terminal':terminal,'checks':checks,'pass':all(checks.values())})
    if not candidate_ok:status='FAIL_CANDIDATE'
    elif not controls_ok:status='NOT_EVALUABLE'
    elif not all(x['pass'] for x in comparisons['R']):status='FAIL_MECHANISM_SCREEN'
    elif not all(x['pass'] for x in comparisons['C']):status='FAIL_VALUE_SCREEN'
    else:status='SCREEN_POSITIVE_HOLD'
    return {'status':status,'summary':summaries,'comparisons':comparisons,'slots':slots,
        'mechanism_screen':'PASS' if candidate_ok and controls_ok and all(x['pass'] for x in comparisons['R']) else ('NOT_EVALUABLE' if not controls_ok and candidate_ok else 'FAIL'),
        'value_screen':('PASS' if status=='SCREEN_POSITIVE_HOLD' else 'FAIL') if candidate_ok and controls_ok and all(x['pass'] for x in comparisons['R']) else 'NOT_INTERPRETED',
        'interpretation':'Fixed-budget operational screening only; no significance, population superiority, biological activity or submission-readiness claim.'}
