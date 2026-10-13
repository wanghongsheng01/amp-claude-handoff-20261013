"""Frozen low-order metrics and conservative complete-slot gates."""
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

def reconstruction(truth, decoded):
    rows=[]
    for i,(target,prediction,status) in enumerate(zip(truth,decoded['sequences'],decoded['status'])):
        if prediction is None:
            rows.append({'row':i,'status':status,'accuracy':None});continue
        matches=sum(i<len(prediction) and aa==prediction[i] for i,aa in enumerate(target))
        rows.append({'row':i,'status':status,'truth':target,'prediction':prediction,'matched_true_AA':matches,
            'true_AA':len(target),'accuracy':matches/len(target),'extra_length':max(0,len(prediction)-len(target)),
            'absolute_length_error':abs(len(prediction)-len(target)),'exact_match':prediction==target})
    complete=all(r['accuracy'] is not None for r in rows)
    return {'rows':rows,'n':len(rows),'complete':complete,
        'accuracy_equal_sequence':sum(r['accuracy'] for r in rows)/len(rows) if complete else None,
        'length_MAE':sum(r['absolute_length_error'] for r in rows)/len(rows) if complete else None,
        'extra_length_mean':sum(r['extra_length'] for r in rows)/len(rows) if complete else None,
        'exact_match_rate':sum(r['exact_match'] for r in rows)/len(rows) if complete else None}

def score_generation(slots,train,dev,check=lambda:None):
    summaries={};nearest={}
    trainseq=[r['sequence'] for r in train]
    trainset=set(trainseq);trainpair={(r['sequence'],r['terminal_state']) for r in train}
    for arm in 'ABCD':
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
                'cap_nonEOS_n':sum(r['status']=='cap_nonEOS' for r in rr) if arm!='D' else None,
                'cap_nonEOS_rate':sum(r['status']=='cap_nonEOS' for r in rr)/16 if arm!='D' else None,
                'conditional_valid_summary':summary,'conditional_valid_vs_dev':comp}
    candidate_ok=all(summaries['A'][t]['main_complete'] for t in ('FREE','AMD'))
    controls_ok=all(summaries[a][t]['main_complete'] for a in 'BCD' for t in ('FREE','AMD'))
    comparisons=[]
    for terminal in ('FREE','AMD'):
        a=summaries['A'][terminal]
        for control in 'BCD':
            b=summaries[control][terminal]
            checks={}
            for metric in ('mmd2_linear_biased_fixed_features','dipeptide_jensen_shannon_bits'):
                x,y=a['conditional_valid_vs_dev'][metric],b['conditional_valid_vs_dev'][metric]
                checks[metric]=x is not None and y is not None and x<y
            for metric in ('length_mean_difference_absolute','copy_train_n','duplicates_rate'):
                x,y=a[metric],b[metric];checks[metric]=x is not None and y is not None and x<=y
            checks['legal5_48_rate']=a['legal5_48_rate']>=b['legal5_48_rate']
            comparisons.append({'terminal':terminal,'control':control,'checks':checks,'pass':all(checks.values())})
    status='FAIL_CANDIDATE' if not candidate_ok else ('NOT_EVALUABLE' if not controls_ok else ('PASS' if all(x['pass'] for x in comparisons) else 'FAIL'))
    return {'status':status,'summary':summaries,'comparisons':comparisons,'slots':slots,
        'interpretation':'Sequence statistics only; no efficacy, novelty, generalization or submission-readiness claim.'}

def information_gate(dev_results):
    def exact_mean(arm,prefix):
        condition_values=[]
        for seed in range(1422,1430):
            result=dev_results.get(f'{arm}_{prefix}{seed}',{}).get('reconstruction',{})
            rows=result.get('rows',[])
            if not result.get('complete') or len(rows)!=41:
                return None
            if any('matched_true_AA' not in r or r.get('true_AA',0)<=0 for r in rows):
                return None
            condition_values.append(sum((Fraction(r['matched_true_AA'],r['true_AA']) for r in rows),Fraction(0))/41)
        return sum(condition_values,Fraction(0))/8
    def ratio_record(v):
        return {'numerator':v.numerator,'denominator':v.denominator} if v is not None else None
    a,b,m=exact_mean('A','S'),exact_mean('B','S'),exact_mean('A','mismatch_S')
    c=dev_results.get('C_plain',{}).get('reconstruction',{}).get('accuracy_equal_sequence')
    complete=all(v is not None for v in (a,b,m))
    ab,am=(a-b,a-m) if complete else (None,None)
    return {'A_S8':float(a) if a is not None else None,'B_S8':float(b) if b is not None else None,
        'A_mismatch_S8':float(m) if m is not None else None,'C':c,
        'A_minus_B':float(ab) if ab is not None else None,'A_minus_mismatch':float(am) if am is not None else None,
        'A_minus_C_descriptive_capacity_unmatched':float(a)-c if a is not None and c is not None else None,
        'exact_rational':{'A':ratio_record(a),'B':ratio_record(b),'A_mismatch':ratio_record(m),
            'A_minus_B':ratio_record(ab),'A_minus_mismatch':ratio_record(am),'threshold':ratio_record(Fraction(1,20))},
        'status':('PASS' if ab>=Fraction(1,20) and am>=Fraction(1,20) else 'FAIL') if complete else 'NOT_EVALUABLE'}
