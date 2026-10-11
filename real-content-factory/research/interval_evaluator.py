"""Independent host oracle and bounded correctness-only interval cases."""
import ast
import hashlib
import itertools
import json
import random

SAFE_BUILTINS = ('sorted','len','min','max','list','tuple','range','enumerate','zip','reversed')


def validate_source(source):
    """A small algorithm subset reduces evaluator spoofing; it is not confinement."""
    if not isinstance(source,str) or not source.strip() or len(source.encode())>4096: raise ValueError('candidate source bound')
    tree=ast.parse(source)
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef): raise ValueError('one algorithm function required')
    function=tree.body[0]
    if function.name!='merge_intervals' or function.decorator_list or function.returns or function.args.defaults or function.args.kw_defaults or function.args.vararg or function.args.kwarg or function.args.kwonlyargs or function.args.posonlyargs or len(function.args.args)!=1 or function.args.args[0].arg!='intervals' or function.args.args[0].annotation:
        raise ValueError('fixed merge_intervals(intervals) signature required')
    forbidden=(ast.Import,ast.ImportFrom,ast.ClassDef,ast.Global,ast.Nonlocal,ast.With,ast.AsyncWith,ast.AsyncFunctionDef,ast.Lambda,ast.Try,ast.Raise,ast.Delete,ast.Yield,ast.YieldFrom,ast.Await)
    for node in ast.walk(tree):
        if isinstance(node,forbidden) or (isinstance(node,ast.FunctionDef) and node is not function): raise ValueError('algorithm subset violation')
        if isinstance(node,ast.Name) and ('__' in node.id or node.id in ('eval','exec','open','globals','locals','getattr','setattr','compile','input','print')): raise ValueError('forbidden algorithm name')
        if isinstance(node,ast.Attribute) and node.attr not in ('append','sort','copy','extend'): raise ValueError('forbidden algorithm attribute')
        if isinstance(node,ast.Call) and not ((isinstance(node.func,ast.Name) and node.func.id in SAFE_BUILTINS) or isinstance(node.func,ast.Attribute)): raise ValueError('forbidden algorithm call')
    return hashlib.sha256(source.encode()).hexdigest()


def reference(intervals):
    """Independent endpoint sweep, intentionally different from sort-and-merge."""
    events={}
    for low,high in intervals:
        if type(low) is not int or type(high) is not int or low>high: raise ValueError('invalid interval')
        if low==high: continue
        events[low]=events.get(low,0)+1; events[high]=events.get(high,0)-1
    active=0; start=None; result=[]
    for point,delta in sorted(events.items()):
        before=active; active+=delta
        if before==0 and active>0:start=point
        if before>0 and active==0:result.append([start,point])
    return result


def cases():
    intervals=[[a,b] for a in range(-2,3) for b in range(a,3)]
    values=[[]]
    for length in (1,2,3): values.extend([list(pair) for pair in group] for group in itertools.combinations_with_replacement(intervals,length))
    values.extend([[[3,5],[1,3],[2,2]],[[0,10],[1,2],[2,8]],[[1,2],[1,2]],[[1000000,1000001],[-1000001,-1000000]]])
    rng=random.Random(20261010)
    for _ in range(64):
        sample=[]
        for _ in range(rng.randrange(0,24)):
            a,b=sorted((rng.randrange(-100,101),rng.randrange(-100,101)));sample.append([a,b])
        values.append(sample)
    return values


def case_identity(values):
    return hashlib.sha256(json.dumps(values,separators=(',',':')).encode()).hexdigest()


def evaluate_outputs(values,outputs):
    if not isinstance(outputs,list) or len(outputs)!=len(values): raise ValueError('candidate output count differs')
    passed=0; failures=[]
    for index,(inputs,output) in enumerate(zip(values,outputs)):
        expected=reference(inputs)
        valid=isinstance(output,list) and len(output)<=len(inputs) and all(isinstance(pair,list) and len(pair)==2 and all(type(n) is int for n in pair) for pair in output)
        if valid and output==expected:passed+=1
        elif len(failures)<8:failures.append({'case':index,'expected':expected,'returned':output})
    return {'tests_passed':passed,'tests_total':len(values),'failures':failures,'cases_sha256':case_identity(values),
            'correct':passed==len(values),'claim_scope':'observed bounded cases; no proof for all inputs'}
