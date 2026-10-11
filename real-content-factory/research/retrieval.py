"""Pluggable retrieval with explicit origin and content identity."""
import hashlib
from .experiments import ExperimentHeld


class RetrievalProvider:
    origin='retrieved'
    def search(self,query,limit=4):
        raise ExperimentHeld('retrieval held: reviewed provider, read scope and request budget required')


class OperatorProvider:
    origin='operator'
    def __init__(self,records): self.records=records
    def search(self,query,limit=4):
        validate_query(query,limit)
        return [evidence(item['id'],item['excerpt'],'operator',None) for item in self.records[:limit]]


def validate_query(query,limit):
    if not isinstance(query,str) or not query.strip() or len(query.encode())>512 or type(limit) is not int or not 1<=limit<=4:raise ValueError('bounded retrieval query required')


def evidence(identity,excerpt,origin,url):
    if origin not in ('operator','mock','retrieved') or not isinstance(identity,str) or not identity or len(identity)>128 or not isinstance(excerpt,str) or not excerpt.strip() or len(excerpt.encode())>512:raise ValueError('invalid retrieval evidence')
    if origin=='retrieved' and (not isinstance(url,str) or not url.startswith('https://')):raise ValueError('retrieved evidence requires source URL')
    if origin!='retrieved' and url is not None:raise ValueError('mock/operator evidence cannot claim retrieval URL')
    return {'id':identity,'excerpt':excerpt,'origin':origin,'url':url,'sha256':hashlib.sha256(excerpt.encode()).hexdigest()}


def retrieve(provider,query,limit=4):
    validate_query(query,limit); records=provider.search(query,limit)
    if not isinstance(records,list) or len(records)>limit:raise ValueError('provider result bound exceeded')
    output=[];ids=set()
    for record in records:
        if not isinstance(record,dict) or set(record)!={'id','excerpt','origin','url','sha256'} or record['origin']!=provider.origin:raise ValueError('provider origin mismatch')
        validated=evidence(record['id'],record['excerpt'],record['origin'],record['url'])
        if validated!=record or record['id'] in ids:raise ValueError('retrieval identity changed/repeated')
        ids.add(record['id']);output.append(record)
    return output
