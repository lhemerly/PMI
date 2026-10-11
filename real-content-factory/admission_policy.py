"""Exact smoke admission: tolerate isolated host paging, stop sustained pressure.

Host counters include unrelated work. There is no owned inference process during
admission. Runtime supervisors and their existing stop thresholds are unchanged.
"""
import json
import math
from pathlib import Path

WINDOW_SECONDS = 2.0
SWAPOUT_RATE_BYTES_PER_SECOND = 1024 * 1024
TOTAL_SWAPOUT_RATE_BYTES_PER_SECOND = 256 * 1024
FULL_PSI_RATIO = .125
SOME_PSI_RATIO = .25


def evaluate(samples):
    if len(samples) < 2:
        raise ValueError('admission counter coverage missing')
    keys = ('swapout', 'PSI_full_us', 'PSI_some_us', 'oom_kill')
    for sample in samples:
        if not isinstance(sample.get('time'), (int,float)) or not math.isfinite(sample['time']):
            raise ValueError('invalid admission timestamp')
        for key in keys:
            if type(sample.get(key)) is not int or sample[key] < 0:
                raise ValueError('missing/invalid admission counter: '+key)
        if type(sample.get('available')) is not int or sample['available'] < 16*1024**3:
            raise ValueError('RAM admission below16GiB')
    for a,b in zip(samples,samples[1:]):
        if not 0 < b['time']-a['time'] <= 1:
            raise ValueError('admission sampling continuity lost')
        if any(b[k] < a[k] for k in keys):
            raise ValueError('admission counter reset/regression')
    first,last = samples[0],samples[-1]
    duration = last['time']-first['time']
    if duration < 30 or duration > 31:
        raise ValueError('admission needs fresh30second coverage')
    if last['oom_kill'] != first['oom_kill']:
        raise ValueError('OOM counter changed')
    peaks = {'full_PSI_ratio':0., 'some_PSI_ratio':0., 'swapout_bytes_per_second':0.}
    # Evaluate overlapping elapsed-time windows, not resetting baselines.
    for i,end in enumerate(samples):
        starts = [s for s in samples[:i] if WINDOW_SECONDS <= end['time']-s['time'] <= WINDOW_SECONDS+1]
        if not starts:
            continue
        start = starts[-1]
        elapsed = end['time']-start['time']
        rates = {'full_PSI_ratio':(end['PSI_full_us']-start['PSI_full_us'])/(elapsed*1e6),
                 'some_PSI_ratio':(end['PSI_some_us']-start['PSI_some_us'])/(elapsed*1e6),
                 'swapout_bytes_per_second':(end['swapout']-start['swapout'])/elapsed}
        for k,v in rates.items(): peaks[k] = max(peaks[k],v)
    total_rate = (last['swapout']-first['swapout'])/duration
    if peaks['full_PSI_ratio'] > FULL_PSI_RATIO or peaks['some_PSI_ratio'] > SOME_PSI_RATIO:
        raise ValueError('sustained host memory PSI during admission')
    if peaks['swapout_bytes_per_second'] > SWAPOUT_RATE_BYTES_PER_SECOND or total_rate > TOTAL_SWAPOUT_RATE_BYTES_PER_SECOND:
        raise ValueError('substantial host paging during admission')
    return {'policy':'bonsai-smoke-admission-window-v1','duration_seconds':duration,
            'scope':'host; no owned inference process exists during admission',
            'swapout_growth_bytes':last['swapout']-first['swapout'],
            'total_swapout_bytes_per_second':total_rate,'peak_windows':peaks,
            'limits':{'window_seconds':2,'full_PSI_ratio':FULL_PSI_RATIO,'some_PSI_ratio':SOME_PSI_RATIO,
                      'swapout_bytes_per_second':SWAPOUT_RATE_BYTES_PER_SECOND,
                      'total_swapout_bytes_per_second':TOTAL_SWAPOUT_RATE_BYTES_PER_SECOND}}


def admit(output, seconds, deadline, check, *, elapsed_coverage=False):
    from window_common import passive
    try:
        passive(output, seconds, deadline, check)
    except ValueError:
        # Replace only reviewed pressure policy and speech elapsed coverage.
        # Collection, deadline, thermal, busy, source and all other failures stay.
        if not Path(output).exists():
            raise
        saved = json.loads(Path(output).read_text())
        allowed = {'passive memory pressure'}
        if elapsed_coverage:
            allowed.add('passive coverage/count/duration')
        if not saved.get('reasons') or not set(saved['reasons']).issubset(allowed):
            raise
    saved = json.loads(Path(output).read_text())
    allowed = {'passive memory pressure'}
    if elapsed_coverage:
        allowed.add('passive coverage/count/duration')
    other = [r for r in saved['reasons'] if r not in allowed]
    if other:
        raise ValueError(', '.join(other))
    if elapsed_coverage and seconds != 30:
        raise ValueError('elapsed speech admission requires30seconds')
    receipt = evaluate(saved['samples'])
    if elapsed_coverage:
        receipt['coverage_policy'] = 'elapsed30to31seconds/max-gap1second'
        receipt['sample_count'] = len(saved['samples'])
        receipt['maximum_sample_gap_seconds'] = max(b['time']-a['time'] for a,b in zip(saved['samples'],saved['samples'][1:]))
    with Path(str(output)+'.policy.json').open('x') as file:
        json.dump(receipt,file,indent=2)


def speech_admit(output, seconds, deadline, check):
    """Speech-only: prove actual elapsed coverage despite integrity-check overhead."""
    return admit(output, seconds, deadline, check, elapsed_coverage=True)


def _elapsed_postflight(output, seconds, deadline, check, *, policy, label, recheck=False):
    """Replace only the passive sample-count rule with proven elapsed coverage."""
    from window_common import passive
    try:
        return passive(output, seconds, deadline, check)
    except ValueError:
        if not Path(output).exists():
            raise
        saved = json.loads(Path(output).read_text())
        if saved.get('reasons') != ['passive coverage/count/duration']:
            raise
        samples = saved['samples']
        if (len(samples) < 2 or any(not math.isfinite(s['time']) for s in samples) or
                not seconds <= samples[-1]['time']-samples[0]['time'] <= seconds+1 or
                any(not 0 < b['time']-a['time'] <= 1 for a,b in zip(samples,samples[1:]))):
            raise ValueError(label + ' postflight elapsed coverage/gaps failed')
        if recheck:
            # Preserve the installed collector's full non-coverage policy even
            # when assessing a saved failed window; no counter is substituted.
            from window_common import passive_reasons
            other = [reason for reason in passive_reasons(samples, seconds)
                     if reason != 'passive coverage/count/duration']
            if other:
                raise ValueError('; '.join(other))
        receipt = {'policy':policy,
                   'duration_seconds':samples[-1]['time']-samples[0]['time'],
                   'sample_count':len(samples),'reasons':[]}
        with Path(str(output)+'.coverage.json').open('x') as file:
            json.dump(receipt,file,indent=2)
        return samples[-1]['time']


def speech_postflight(output, seconds, deadline, check):
    """Same speech release policy and receipt, factoring elapsed coverage only."""
    return _elapsed_postflight(output, seconds, deadline, check,
                              policy='speech-postflight-elapsed-coverage-v1', label='speech')


def demo_postflight(output, seconds, deadline, check):
    """Reviewed model demo only: 20–21 elapsed seconds, every other check intact."""
    if type(seconds) is not int or seconds != 20:
        raise ValueError('model demo postflight requires exactly20seconds')
    return _elapsed_postflight(output, seconds, deadline, check,
                              policy='model-demo-postflight-elapsed-coverage-v1',
                              label='model demo', recheck=True)
