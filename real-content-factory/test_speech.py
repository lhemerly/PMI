"""Model-free contract fixtures; WAV fixtures are silence, never inference."""
import hashlib
import fcntl
import json
from pathlib import Path
import struct
import tempfile
import time
import unittest
from unittest.mock import patch
import wave

import content_factory as factory
from launch_policy import PlatformHeld
from llama_backend import require_prompt_fit
from piper_backend import PiperBackend, check_speech_text, speech_guarded_argv, validate_wav
import piper_worker
from prompt_fit import fit_prompt, gguf_metadata
from test_contract import ContractBackend


class SpeechFixture:
    actual_inference = False

    def __init__(self, text_backend, release=True):
        self.text_backend = text_backend
        self.release = release
        self.called = False

    def run_stage(self, text, stage, deadline):
        if self.text_backend.live:
            raise AssertionError('text/speech overlap')
        self.called = True
        self.text_backend.events.append(('speech', text))
        with wave.open(str(stage / 'answer.wav'), 'wb') as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(22050)
            wav.writeframes(b'\0\0' * 100)
        return {'returncode': 0, 'owned_cleanup_empty': self.release, 'postflight_passed': self.release}


class KokoroSpeechFixture:
    """Silence-only 24 kHz contract fixture; never imports or invokes Kokoro."""
    actual_inference = False
    model_id = 'kokoro-82m'

    def __init__(self, text_backend, voice_id='af_heart'):
        self.text_backend = text_backend
        self.voice_id = voice_id
        self.request = None

    def run_stage(self, text, stage, deadline):
        if self.text_backend.live:
            raise AssertionError('text/speech overlap')
        self.request = json.loads((stage / 'request.json').read_text())
        with wave.open(str(stage / 'answer.wav'), 'wb') as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(24000)
            wav.writeframes(b'\0\0' * 100)
        return {'returncode': 0, 'owned_cleanup_empty': True, 'postflight_passed': True,
                'runtime_version': '0.9.4',
                'runtime_code_status': 'mocked-runtime-boundary',
                'runtime_qualification': 'unqualified',
                'model_repository': 'hexgrad/Kokoro-82M',
                'model_revision': 'f3ff3571791e39611d31c381e3a41a3af07b4987',
                'model_sha256': None, 'model_sha256_verified': False,
                'model_config_sha256': None, 'voice_pack_sha256': None,
                'input_seal_sha256': None}


class Tests(unittest.TestCase):
    def setUp(self):
        # Missing-asset fixtures must not depend on the host's installed runtime.
        fixture = tempfile.TemporaryDirectory()
        self.addCleanup(fixture.cleanup)
        base = Path(fixture.name)
        isolated = patch.multiple('kokoro_backend',
            MODEL_FILE=base/'kokoro-v1_0.pth', MODEL_CONFIG_FILE=base/'config.json',
            VOICE_DIRECTORY=base/'voices', RUNTIME_DIRECTORY=base/'runtime',
            INPUT_SEAL=base/'input-seal.json')
        isolated.start()
        self.addCleanup(isolated.stop)

    def test_exact_kokoro_authorization_preserves_other_holds(self):
        from launch_policy import require_approved_kokoro_sample, require_real_execution
        quote = 'O enxame é o instrumento da minha vontade.'
        require_approved_kokoro_sample(quote, 'pf_dora')
        for text, voice in ((quote, 'af_heart'), ('olá', 'pf_dora')):
            with self.assertRaises(PlatformHeld):
                require_approved_kokoro_sample(text, voice)
        with self.assertRaises(PlatformHeld):
            require_real_execution()

    def test_speech_elapsed_coverage_handles_overhead_and_keeps_all_guards(self):
        import copy, sys
        from admission_policy import speech_admit
        from llama_backend import QUAL
        sys.path.insert(0,str(QUAL))
        import window_common
        source = factory.HERE.parent/'first-bonsai-speech-job-006/speech/admission.json'
        samples = json.loads(source.read_text())['samples']
        self.assertEqual(len(samples),59)
        def exercise(rows, passes):
            with tempfile.TemporaryDirectory() as temp:
                output=Path(temp)/'admission.json'
                def original(output,seconds,deadline,check):
                    reasons=window_common.passive_reasons(rows,seconds)
                    output.write_text(json.dumps({'samples':rows,'reasons':reasons}))
                    if reasons:raise ValueError('; '.join(reasons))
                with patch('window_common.passive',side_effect=original):
                    if passes:
                        speech_admit(output,30,100,lambda:None)
                        receipt=json.loads(Path(str(output)+'.policy.json').read_text())
                        self.assertEqual(receipt['sample_count'],59)
                    else:
                        with self.assertRaises(ValueError):speech_admit(output,30,100,lambda:None)
        exercise(samples,True)
        for kind in ['gap','short','thermal','RAM','VRAM','busy','OOM','source','reset','PSI','swap']:
            rows=copy.deepcopy(samples)
            if kind=='gap':rows[1]['time']=rows[0]['time']+1.1
            if kind=='short':
                first=rows[0]['time']
                for r in rows:r['time']=first+(r['time']-first)*.9
            if kind=='thermal':rows[20]['CPU_package_mC']=80001
            if kind=='RAM':rows[20]['available']=16*1024**3-1
            if kind=='VRAM':rows[20]['vram']=1024**3+1
            if kind=='busy':rows[20]['busy']=11
            if kind=='OOM':rows[-1]['oom_kill']+=1
            if kind=='source':rows[20]['source_identity']={'different':True}
            if kind=='reset':rows[20]['swapout']=0
            if kind=='PSI':
                first=rows[0]
                for r in rows:r['PSI_full_us']=first['PSI_full_us']+int((r['time']-first['time'])*200000)
            if kind=='swap':
                first=rows[0]
                for r in rows:r['swapout']=first['swapout']+int((r['time']-first['time'])*2*1024**2)
            with self.subTest(kind=kind):exercise(rows,False)

    def test_speech_postflight_preserves_pressure_and_coverage_checks(self):
        from admission_policy import speech_postflight
        import window_common
        rows=json.loads((factory.HERE.parent/'first-bonsai-speech-job-006/speech/admission.json').read_text())['samples'][:39]
        start=rows[0]['time']
        for i,row in enumerate(rows):row['time']=start+i*(20.3/38)
        with tempfile.TemporaryDirectory() as temp:
            def exercise(reasons,expected):
                output=Path(temp)/str(len(list(Path(temp).iterdir())))
                def original(*args):
                    output.write_text(json.dumps({'samples':rows,'reasons':reasons}))
                    raise ValueError('; '.join(reasons))
                with patch('window_common.passive',side_effect=original):
                    if expected:speech_postflight(output,20,100,lambda:None)
                    else:
                        with self.assertRaises(ValueError):speech_postflight(output,20,100,lambda:None)
            exercise(window_common.passive_reasons(rows,20),True)
            exercise(['passive coverage/count/duration','passive memory pressure'],False)
            rows[-1]['time']=start+19
            exercise(['passive coverage/count/duration'],False)

    def test_verified_resume_requires_sealed_answer_and_completion_release(self):
        import piper_backend as piper
        from prompt_fit import SMOKE_PROMPT_SHA256
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);job=base/'source';stage=job/'stage-0-bonsai-stock';stage.mkdir(parents=True)
            prompt=(factory.HERE/'first-run-prompt.txt').read_bytes();answer=b'Verified welcome.'
            (job/'prompt.txt').write_bytes(prompt);(job/'plan.json').write_text(json.dumps({'stages':[{'model_id':'bonsai-stock'}]}))
            (stage/'request.json').write_text(json.dumps({'model_id':'bonsai-stock','prompt':prompt.decode()}));(stage/'answer.txt').write_bytes(answer+b'\n')
            (stage/'native.stdout.log').write_bytes(answer+b' [end of text]\n');(stage/'native.stderr.log').write_text('native evidence')
            result={'status':'RealTextAndReleasePass','actual_inference':True,'model_id':'bonsai-stock','owned_cleanup_empty':True,'postflight_passed':True,'answer_sha256':hashlib.sha256(answer).hexdigest()}
            (stage/'result.json').write_text(json.dumps(result));(stage/'placement.json').write_text('{}')
            host={'event':'final','status':'completed','command_returncode':0,'remaining_live_pids':[],'monitor_errors':[],'command_started':True}
            gpu={'event':'final','status':'completed','command_returncode':0,'cleanup_incomplete':False,'errors':[]}
            (stage/'host.jsonl').write_text(json.dumps(host)+'\n');(stage/'vram.jsonl').write_text(json.dumps(gpu)+'\n');(stage/'postflight.json').write_text('{"reasons":[]}')
            paths=[job/'prompt.txt',job/'plan.json',*sorted(stage.iterdir())]
            manifest={'schema':'PMI_VERIFIED_TEXT_RESUME_V1','source_job':str(job),'model_id':'bonsai-stock','model_sha256':factory.MODELS['bonsai-stock']['sha256'],'prompt_sha256':hashlib.sha256(prompt).hexdigest(),'answer':answer.decode(),'answer_sha256':hashlib.sha256(answer).hexdigest(),'answer_file_sha256':hashlib.sha256(answer+b'\n').hexdigest(),'files_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
            m=base/'resume-speech-provenance.json';m.write_text(json.dumps(manifest))
            files={**manifest['files_sha256'],str(m):hashlib.sha256(m.read_bytes()).hexdigest()}
            seal=base/'speech-input-seal.json';seal.write_text(json.dumps({'schema':'PMI_PIPER_CPU_INTEGRITY_V1','files_sha256':files}))
            with patch('piper_backend.HERE',base):
                text,actual,_=piper.verified_resume_text(job,time.monotonic()+5)
                self.assertEqual(text,answer.decode());self.assertEqual(hashlib.sha256(actual.encode()).hexdigest(),SMOKE_PROMPT_SHA256)
                with self.assertRaises(ValueError):piper.verified_resume_text(base/'different',time.monotonic()+5)
                (stage/'answer.txt').write_text('Changed answer')
                with self.assertRaises(ValueError):piper.verified_resume_text(job,time.monotonic()+5)
                (stage/'answer.txt').write_bytes(answer+b'\n')
                manifest['files_sha256'].pop(str(stage/'host.jsonl'));m.write_text(json.dumps(manifest));files[str(m)]=hashlib.sha256(m.read_bytes()).hexdigest();seal.write_text(json.dumps({'schema':'PMI_PIPER_CPU_INTEGRITY_V1','files_sha256':files}))
                with self.assertRaises(ValueError):piper.verified_resume_text(job,time.monotonic()+5)

    def test_resume_speech_never_constructs_text_backend_and_keeps_lease(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);text=ContractBackend();speech=SpeechFixture(text)
            def verified(source,deadline):return 'Verified answer.', 'Pinned prompt', {'source_job':str(source)}
            with patch('piper_backend.verified_resume_text',side_effect=verified), patch('llama_backend.LlamaBackend',side_effect=AssertionError('text backend')):
                result=factory._resume_speech(base/'source',base/'out',base/'lease',speech)
                self.assertFalse(result['text_model_launch']);self.assertEqual(text.events,[('speech','Verified answer.')])
                self.assertEqual(result['status'],'ContractOnly')
                with (base/'lease').open('rb') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    with self.assertRaises(BlockingIOError):factory._resume_speech(base/'source',base/'busy',base/'lease',speech)
                self.assertFalse((base/'busy').exists())

    def test_resume_speech_can_select_kokoro_without_rewriting_source_receipts(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); text = ContractBackend()
            speech = KokoroSpeechFixture(text, 'pf_dora')
            original_source = {'job': str(base/'source'), 'answer': 'Verified answer.'}
            def verified(source, deadline):
                self.assertEqual(str(source), original_source['job'])
                return original_source['answer'], 'Pinned prompt', {'source_job': str(source)}
            with patch('piper_backend.verified_resume_text', side_effect=verified), \
                 patch('llama_backend.LlamaBackend', side_effect=AssertionError('text backend')):
                result = factory._resume_speech(base/'source', base/'out', base/'lease', speech)
            self.assertFalse(result['text_model_launch'])
            self.assertEqual(result['speech_model'], 'kokoro-82m')
            self.assertEqual(result['voice_id'], 'pf_dora')
            self.assertEqual(result['voice_language'], 'pt-BR')
            self.assertEqual(result['sample_rate'], 24000)
            self.assertEqual(result['status'], 'ContractOnly')
            self.assertFalse(result['actual_inference'])
            self.assertEqual(result['runtime_code_status'], 'mocked-runtime-boundary')
            self.assertEqual(result['runtime_qualification'], 'unqualified')
            self.assertEqual(result['model_repository'], 'hexgrad/Kokoro-82M')
            self.assertFalse(result['model_sha256_verified'])
            self.assertIsNone(result['model_sha256'])
            self.assertEqual(result['asset_identity_status'], 'mock-unverified')
            request = json.loads((base/'out/speech/request.json').read_text())
            self.assertEqual(request['model_id'], 'kokoro-82m')
            self.assertEqual(request['voice_id'], 'pf_dora')
            self.assertEqual(original_source['answer'], 'Verified answer.')

    def test_kokoro_capture_rejects_unverified_real_inference_identity(self):
        import kokoro_backend as kokoro
        from content_factory import _kokoro_capture_metadata

        capture = {'runtime_version': kokoro.RUNTIME_VERSION,
                   'runtime_code_status': 'mocked-runtime-boundary',
                   'runtime_qualification': 'unqualified',
                   'model_repository': kokoro.KOKORO_REPOSITORY,
                   'model_revision': kokoro.KOKORO_REVISION,
                   'model_sha256': None, 'model_sha256_verified': False,
                   'model_config_sha256': None, 'voice_pack_sha256': None,
                   'input_seal_sha256': None}
        metadata = _kokoro_capture_metadata(capture, actual_inference=False)
        self.assertEqual(metadata['asset_identity_status'], 'mock-unverified')
        self.assertFalse(metadata['model_sha256_verified'])
        with self.assertRaisesRegex(RuntimeError, 'verified pinned input hashes'):
            _kokoro_capture_metadata(capture, actual_inference=True)
        verified_capture = {**capture,
                            'runtime_code_status': 'executed-this-receipt',
                            'model_sha256': kokoro.KOKORO_SHA256,
                            'model_sha256_verified': True,
                            'model_config_sha256': 'a' * 64,
                            'voice_pack_sha256': 'b' * 64,
                            'input_seal_sha256': 'c' * 64}
        verified_metadata = _kokoro_capture_metadata(verified_capture, actual_inference=True)
        self.assertEqual(verified_metadata['asset_identity_status'], 'verified-before-launch')
        self.assertTrue(verified_metadata['model_sha256_verified'])

    def test_resume_entrypoint_held_before_provenance_or_outputs(self):
        with tempfile.TemporaryDirectory() as temp, patch('subprocess.Popen',side_effect=AssertionError('launch')):
            output=Path(temp)/'absent'
            with self.assertRaises(PlatformHeld):factory.main(['resume-speech','--resume-from',str(output),'--out',str(output)])
            self.assertFalse(output.exists())

    def test_sealed_ljspeech_voice_configuration_uses_its_english_espeak_id(self):
        config=json.loads((factory.HERE/'voices/en_US-ljspeech-medium.onnx.json').read_text())
        self.assertEqual(config['phoneme_type'],'espeak')
        self.assertEqual(config['espeak']['voice'],'en')
        self.assertEqual(config['language']['code'],'en_US')
        self.assertTrue(piper_worker.validate_voice_config(config))
        for mutate in [
            lambda c:c['espeak'].__setitem__('voice','en-us'),
            lambda c:c['espeak'].__setitem__('voice','en-gb'),
            lambda c:c.__setitem__('phoneme_type','text'),
            lambda c:c['language'].__setitem__('code','en_GB'),
            lambda c:c.pop('language'),
        ]:
            changed=json.loads(json.dumps(config));mutate(changed)
            with self.assertRaises(ValueError):piper_worker.validate_voice_config(changed)
        with self.assertRaises(ValueError):piper_worker.validate_voice_config(None)

    def test_failed_integrated_speech_stage_keeps_resumeable_completed_text_and_progress(self):
        import contextlib,io
        class RealProofBackend:
            actual_inference=True
            def run_stage(self,key,prompt,stage,deadline):
                raw=b'Welcome, listener. [end of text]\n'
                (stage/'native.stdout.log').write_bytes(raw);(stage/'native.stderr.log').write_text('')
                (stage/'placement.json').write_text('{}')
                (stage/'host.jsonl').write_text(json.dumps({'event':'final','status':'completed','command_returncode':0,'command_started':True,'remaining_live_pids':[],'monitor_errors':[]})+'\n')
                (stage/'vram.jsonl').write_text(json.dumps({'event':'final','status':'completed','command_returncode':0,'cleanup_incomplete':False,'errors':[]})+'\n')
                (stage/'postflight.json').write_text('{"reasons":[]}')
                return {'returncode':0,'owned_cleanup_empty':True,'postflight_passed':True,'raw_stdout':raw}
        class FailedSpeech:
            actual_inference=False
            def run_stage(self,*args):raise RuntimeError('fixture speech failure')
        prompt=(factory.HERE/'first-run-prompt.txt').read_text()
        with tempfile.TemporaryDirectory() as temp, patch('content_factory.require_real_execution'):
            base=Path(temp);buffer=io.StringIO()
            with contextlib.redirect_stdout(buffer),self.assertRaisesRegex(RuntimeError,'fixture speech failure'):
                factory._run_sequence(prompt,['bonsai-stock'],base/'job',base/'lease',RealProofBackend(),FailedSpeech())
            saved=base/'job';status=json.loads((saved/'result.json').read_text())
            self.assertEqual(status['status'],'StoppedNoRetry')
            self.assertTrue(status['paths']['resume_provenance'].endswith('resume-speech-provenance.json'))
            text,used_prompt,_=__import__('piper_backend').verified_resume_text(saved,time.monotonic()+10)
            self.assertEqual(text,'Welcome, listener.');self.assertEqual(used_prompt,prompt)
            events=[json.loads(line) for line in (saved/'progress.jsonl').read_text().splitlines()]
            self.assertEqual([(e['stage'],e['status']) for e in events],[('text:bonsai-stock','running'),('text:bonsai-stock','complete'),('speech:piper-cpu','running'),('speech:piper-cpu','failed')])
            self.assertIn('[speech:piper-cpu] failed',buffer.getvalue())

    def test_image_stage_runs_after_text_release_and_speech_under_one_lease(self):
        import contextlib, io
        import comfy_backend as comfy
        from test_contract import FakeComfy
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); text = ContractBackend(); speech = SpeechFixture(text)
            image = FakeComfy()
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                result = factory._run_sequence('A quiet harbour at dawn', ['qwen3-stock'],
                                               base / 'job', base / 'lease', text, speech, image)
            # Ordering is the contract: text unloads, speech runs, then the image submits.
            self.assertEqual(text.events[-3:], [('load', 'qwen3-stock'), ('exit-and-release', 'qwen3-stock'),
                                               ('speech', 'contract text')])
            self.assertEqual([call[1] for call in image.calls][:2], ['/system_stats', '/prompt'])
            self.assertEqual(result['status'], 'ContractOnly')
            self.assertFalse(result['actual_inference'])
            self.assertEqual(result['paths']['wav'], str(base / 'job/speech/answer.wav'))
            self.assertEqual(result['paths']['image'], str(base / 'job/image/answer.png'))
            self.assertEqual(result['paths']['image_receipt'], str(base / 'job/image/result.json'))
            self.assertEqual([row.get('modality') for row in result['stages']], [None, None, 'image'])
            self.assertEqual(result['stages'][2]['status'], 'ContractOnly')
            self.assertEqual(result['stages'][2]['endpoint'], 'http://127.0.0.1:8188')
            saved = json.loads((base / 'job/image/result.json').read_text())
            self.assertEqual(saved['status'], 'ContractOnly')
            self.assertFalse(saved['actual_inference'])
            self.assertTrue(saved['release_proven'])
            self.assertFalse(saved['automatic_retry'])
            self.assertEqual(saved['width'], 512)
            self.assertEqual(saved['sha256'],
                             hashlib.sha256((base / 'job/image/answer.png').read_bytes()).hexdigest())
            self.assertEqual([json.loads(line)['stage'] for line in
                              Path(result['paths']['progress']).read_text().splitlines()],
                             ['text:qwen3-stock', 'text:qwen3-stock', 'speech:piper-cpu',
                              'speech:piper-cpu', 'image:comfyui-local', 'image:comfyui-local'])
            self.assertIn('[image:comfyui-local] complete', buffer.getvalue())
            request = json.loads((base / 'job/image/request.json').read_text())
            self.assertEqual(request['prompt'], 'contract text')
            self.assertEqual(request['endpoint'], 'http://127.0.0.1:8188')
            self.assertTrue((base / 'job/image/workflow.json').is_file())

    def test_image_failure_keeps_completed_text_and_speech_and_never_retries(self):
        import comfy_backend as comfy
        from test_contract import FakeComfy
        for why, options in [('queue error', {'history': [{'status': {'status_str': 'error',
                                                                       'completed': False}}]}),
                             ('unproven unload', {'unload_fails': True}),
                             ('not an image', {'artifact': b'not an image at all'})]:
            with self.subTest(why=why), tempfile.TemporaryDirectory() as temp:
                base = Path(temp); text = ContractBackend(); speech = SpeechFixture(text)
                image = FakeComfy(**options)
                with self.assertRaises((comfy.ComfyFailure, ValueError, RuntimeError)):
                    factory._run_sequence('A quiet harbour at dawn', ['qwen3-stock'], base / 'job',
                                          base / 'lease', text, speech, image)
                saved = json.loads((base / 'job/result.json').read_text())
                self.assertEqual(saved['status'], 'StoppedNoRetry')
                self.assertFalse(saved['actual_inference'])
                self.assertTrue(saved['later_models_held'])
                self.assertFalse(saved['automatic_retry'])
                self.assertEqual([row.get('modality') for row in saved['completed_stages']], [None, None])
                # Completed text and speech artifacts remain on disk for review.
                self.assertTrue((base / 'job/speech/answer.wav').is_file())
                self.assertTrue(Path(saved['paths']['answer']).is_file())
                self.assertEqual(saved['paths']['image'], None)
                self.assertEqual([json.loads(line)['status'] for line in
                                  (base / 'job/progress.jsonl').read_text().splitlines()][-2:],
                                 ['running', 'failed'])
                self.assertNotIn(('POST', '/interrupt', None), image.calls)
                self.assertIn(('POST', '/interrupt', {'prompt_id': '12345678-1234-5678-1234-567812345678'}),
                              image.calls)
                # The shared lease still covers image work: a busy lease refuses the whole job.
                other = ContractBackend()
                with (base / 'lease').open('rb') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    with self.assertRaises(BlockingIOError):
                        factory._run_sequence('A quiet harbour at dawn', ['qwen3-stock'], base / 'busy',
                                              base / 'lease', other, SpeechFixture(other), FakeComfy())
                self.assertEqual(other.events, [])
                self.assertEqual(FakeComfy().calls, [])

    def test_image_stage_never_overlaps_a_live_text_model_or_speech(self):
        from test_contract import FakeComfy
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); text = ContractBackend(); speech = SpeechFixture(text)
            seen = []
            original = FakeComfy.run_stage

            def watched(self, prompt, stage, deadline):
                # Image submits only after text release and after the speech WAV exists.
                seen.append((text.live, speech.called, (stage.parent / 'speech/answer.wav').exists()))
                return original(self, prompt, stage, deadline)

            with patch('test_contract.FakeComfy.run_stage', watched):
                factory._run_sequence('Prompt', ['qwen3-stock'], base / 'job', base / 'lease',
                                      text, speech, FakeComfy())
            self.assertEqual(seen, [(False, True, True)])
            self.assertEqual(text.events[-2:], [('exit-and-release', 'qwen3-stock'),
                                                ('speech', 'contract text')])

    def test_text_to_piper_speech_is_unchanged_without_an_image_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); text = ContractBackend(); speech = SpeechFixture(text)
            result = factory._run_sequence('Prompt', ['qwen3-stock'], base / 'job', base / 'lease',
                                           text, speech)
            self.assertEqual(result['paths']['wav'], str(base / 'job/speech/answer.wav'))
            self.assertIsNone(result['paths']['image'])
            self.assertIsNone(result['paths']['image_receipt'])
            self.assertEqual(result['status'], 'ContractOnly')
            self.assertFalse((base / 'job/image').exists())
            self.assertEqual([row.get('modality') for row in result['stages']], [None, None])
            # Speech-only resume is untouched: it never gains an image path or stage.
            def verified(source, deadline):
                return 'Verified answer.', 'Pinned prompt', {'source_job': str(source)}
            with patch('piper_backend.verified_resume_text', side_effect=verified):
                resume = factory._resume_speech(base / 'job', base / 'resume', base / 'lease', speech)
            self.assertEqual(resume['paths']['wav'], str(base / 'resume/speech/answer.wav'))
            self.assertNotIn('image', resume['paths'])
            self.assertFalse((base / 'resume/image').exists())

    def test_contract_job_paths_and_stage_status_are_obvious(self):
        import contextlib,io
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);buffer=io.StringIO()
            with contextlib.redirect_stdout(buffer):
                result=factory._run_sequence('Short prompt',['qwen3-stock'],base/'job',base/'lease',ContractBackend())
            self.assertEqual(result['status'],'ContractOnly')
            self.assertEqual(result['stages'][0]['status'],'ContractOnly')
            self.assertTrue(Path(result['paths']['answer']).is_file())
            self.assertEqual(result['paths']['wav'],None)
            self.assertEqual([json.loads(x)['status'] for x in Path(result['paths']['progress']).read_text().splitlines()],['running','complete'])

    def test_same_lease_covers_text_release_and_speech_saved_wav(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); text = ContractBackend(); speech = SpeechFixture(text)
            result = factory._run_sequence('Arbitrary user prompt é', ['qwen3-stock'], base/'job', base/'lease', text, speech)
            self.assertEqual(text.events[-2:], [('exit-and-release', 'qwen3-stock'), ('speech', 'contract text')])
            self.assertEqual(result['status'], 'ContractOnly')
            self.assertFalse(result['actual_inference'])
            row = json.loads((base/'job/speech/result.json').read_text())
            self.assertEqual(row['frames'], 100)
            self.assertFalse(row['actual_inference'])

    def test_incomplete_text_or_release_failure_prevents_speech(self):
        for text in (ContractBackend(release=False), ContractBackend(incomplete=True)):
            with tempfile.TemporaryDirectory() as temp:
                base = Path(temp); speech = SpeechFixture(text)
                with self.assertRaises((ValueError, RuntimeError)):
                    factory._run_sequence('Prompt', ['qwen3-stock'], base/'job', base/'lease', text, speech)
                self.assertFalse(speech.called)

    def test_speech_failure_records_no_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); text = ContractBackend(); speech = SpeechFixture(text, False)
            with self.assertRaises(RuntimeError):
                factory._run_sequence('Prompt', ['qwen3-stock'], base/'job', base/'lease', text, speech)
            self.assertEqual(json.loads((base/'job/result.json').read_text())['status'], 'StoppedNoRetry')

    def test_all_speech_physical_entrypoints_held_without_outputs(self):
        with tempfile.TemporaryDirectory() as temp, patch('subprocess.Popen', side_effect=AssertionError('launch')):
            output = Path(temp)/'absent'
            with self.assertRaises(PlatformHeld):
                factory.main(['speak', '--out', str(output)])
            with self.assertRaises(PlatformHeld):
                PiperBackend().run_stage('hello', output, 0)
            with self.assertRaises(PlatformHeld):
                piper_worker.main()
            self.assertFalse(output.exists())

    def test_kokoro_audio_contract_is_24k_mono_and_separate_from_the_piper_baseline(self):
        import io as _io
        import kokoro_backend as kokoro
        import piper_backend as piper

        def wav_bytes(rate, channels, width, frames):
            buffer = _io.BytesIO()
            with wave.open(buffer, 'wb') as wav:
                wav.setnchannels(channels); wav.setsampwidth(width); wav.setframerate(rate)
                wav.writeframes(b'\0\0' * frames * channels)
            return buffer.getvalue()

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'answer.wav'
            path.write_bytes(wav_bytes(24000, 1, 2, 2400))
            receipt = kokoro.validate_wav(path)
            self.assertEqual((receipt['sample_rate'], receipt['channels'], receipt['sample_width_bytes']),
                             (24000, 1, 2))
            self.assertEqual(receipt['frames'], 2400)
            self.assertEqual(receipt['adapter'], 'kokoro-cpu')
            # Same bytes produce the same receipt: no wall-clock or random input.
            self.assertEqual(kokoro.validate_wav(path), receipt)
            # Each validator refuses the other's rate; neither borrows the other's proof.
            self.assertEqual((piper.MAX_WAV, kokoro.MAX_WAV),
                             (22050 * 2 * 30 + 4096, 24000 * 2 * 30 + 4096))
            piper_wav = wav_bytes(22050, 1, 2, 2205)
            path.write_bytes(piper_wav)
            self.assertEqual(piper.validate_wav(path)['sample_rate'], 22050)
            with self.assertRaisesRegex(ValueError, '24000 Hz'):
                kokoro.validate_wav(path)
            path.write_bytes(wav_bytes(24000, 1, 2, 2400))
            self.assertEqual(kokoro.validate_wav(path)['sample_rate'], 24000)
            with self.assertRaisesRegex(ValueError, 'unexpected PCM WAV format'):
                piper.validate_wav(path)
            for why, kwargs in [('stereo', (24000, 2, 2, 2400)), ('16-bit rate drift', (16000, 1, 2, 1600)),
                                ('8-bit width', (24000, 1, 1, 4800)), ('empty', (24000, 1, 2, 0)),
                                ('over duration', (24000, 1, 2, 24000 * 30 + 1))]:
                with self.subTest(why=why):
                    path.write_bytes(wav_bytes(*kwargs))
                    with self.assertRaises(ValueError):
                        kokoro.validate_wav(path)
            for why, data in [('empty file', b''), ('not a WAV', b'RIFF'),
                              ('truncated header', wav_bytes(24000, 1, 2, 10)[:8])]:
                with self.subTest(why=why):
                    path.write_bytes(data)
                    with self.assertRaises((ValueError, EOFError, wave.Error)):
                        kokoro.validate_wav(path)

    def test_kokoro_plan_is_offline_and_marks_assets_absent_and_runtime_unqualified(self):
        import kokoro_backend as kokoro
        with patch('subprocess.Popen', side_effect=AssertionError('process attempted')), \
             patch('socket.socket', side_effect=AssertionError('socket attempted')):
            plan = kokoro.kokoro_plan()
        self.assertEqual(plan['schema'], 'PMI_KOKORO_CPU_SPEECH_V1')
        self.assertEqual(plan['status'], 'PlatformHeld')
        self.assertEqual(plan['adapter'], 'kokoro-cpu')
        self.assertEqual(plan['model_id'], 'kokoro-82m')
        self.assertEqual(plan['device'], 'CPU')
        self.assertFalse(plan['actual_inference'])
        self.assertTrue(plan['wired_into_pipeline'])
        # The metadata-only plan never claims the weights/runtime are ready or hashes them.
        self.assertFalse(plan['model_present'])
        self.assertFalse(plan['assets_present'])
        self.assertFalse(plan['readiness']['ready'])
        self.assertFalse(plan['readiness']['identity_verified'])
        self.assertFalse(plan['runtime_qualified'])
        self.assertTrue(plan['synthesis_implemented'])
        self.assertEqual(plan['runtime_code_status'], 'implemented-but-unrun')
        self.assertEqual(plan['runtime']['python_version'], '>=3.10,<3.13')
        self.assertIn('torch (CPU build)', plan['runtime']['required_packages'])
        self.assertTrue(plan['execution_held'])
        self.assertFalse(plan['assets_downloaded'])
        missing_roles = {entry['role'] for entry in plan['missing_artifacts']
                         if not entry['present']}
        self.assertIn('model_config', missing_roles)
        self.assertIn('model_weights', missing_roles)
        self.assertIn('input_seal', missing_roles)
        self.assertIsNone(plan['model']['bytes'])
        self.assertEqual(plan['model']['license'], 'Apache-2.0')
        self.assertFalse(plan['model']['sha256_verified_here'])
        self.assertEqual(plan['model']['repository'], 'hexgrad/Kokoro-82M')
        # Piper stays the preserved CPU baseline; Kokoro is an additional candidate.
        self.assertTrue(plan['preserved_baseline']['unchanged'])
        self.assertNotIn('video', plan['preserved_baseline'])
        self.assertIn('video', plan['unavailable_modalities'])

    def test_kokoro_voice_selection_is_configurable_and_pinned_by_identity(self):
        import kokoro_backend as kokoro
        from model_intent import KOKORO_REVISION, KOKORO_SHA256
        # Both intended voices are configured, each bound to its own language.
        self.assertEqual(sorted(kokoro.VOICES), ['af_heart', 'pf_dora'])
        self.assertEqual(kokoro.VOICES['af_heart']['language'], 'en-US')
        self.assertEqual(kokoro.VOICES['af_heart']['language_name'], 'English')
        self.assertEqual(kokoro.VOICES['pf_dora']['language'], 'pt-BR')
        self.assertEqual(kokoro.VOICES['pf_dora']['language_name'], 'Brazilian Portuguese')
        self.assertEqual(kokoro.DEFAULT_VOICE, 'af_heart')
        # Configuration is selectable at construction and never implicit.
        self.assertEqual(kokoro.KokoroBackend().voice_id, 'af_heart')
        self.assertEqual(kokoro.KokoroBackend('pf_dora').voice_id, 'pf_dora')
        self.assertEqual(kokoro.KokoroBackend(environ={'PMI_KOKORO_VOICE': 'pf_dora'}).voice_id, 'pf_dora')
        self.assertEqual(kokoro.resolve_voice(None, {}), 'af_heart')
        self.assertEqual(kokoro.resolve_voice(None, {'PMI_KOKORO_VOICE': 'pf_dora'}), 'pf_dora')
        self.assertEqual(kokoro.resolve_voice('af_heart', {'PMI_KOKORO_VOICE': 'pf_dora'}), 'af_heart')
        # An unknown or malformed voice is refused rather than silently defaulted.
        for bad in ['', 'AF_HEART', 'af_herbert', 'zz_dora', 'af_heart ', 7, None, b'af_heart']:
            with self.subTest(voice=bad):
                with self.assertRaises(ValueError):
                    kokoro.check_voice(bad)
        with self.assertRaises(ValueError):
            kokoro.resolve_voice('', {})
        for bad in ['AF_HEART', 'af_herbert', 'zz_dora', 'af_heart ', 7, True]:
            with self.subTest(voice=bad):
                with self.assertRaises(ValueError):
                    kokoro.resolve_voice(bad, {})
                with self.assertRaises(ValueError):
                    kokoro.resolve_voice(None, {'PMI_KOKORO_VOICE': bad})
        with self.assertRaises(ValueError):
            kokoro.resolve_voice(None, {'PMI_KOKORO_VOICE': ''})
        self.assertEqual(kokoro.kokoro_plan('pf_dora')['voice_id'], 'pf_dora')
        self.assertEqual(kokoro.kokoro_plan('pf_dora')['model']['sha256'], KOKORO_SHA256)
        self.assertEqual(kokoro.kokoro_plan('pf_dora')['model']['revision'], KOKORO_REVISION)

    def test_kokoro_text_bound_and_voice_pack_contract(self):
        import kokoro_backend as kokoro
        for bad in ['', ' ', '\0', 'é' * 2049, None, 7]:
            with self.subTest(text=repr(bad)):
                with self.assertRaises(ValueError):
                    kokoro.check_text(bad)
        kokoro.check_text('Bem-vindo ao nosso estúdio local.')
        # The voice pack is a declared path only; nothing here creates or hashes it.
        pack = kokoro.voice_pack('pf_dora')
        self.assertEqual(pack.name, 'pf_dora.pt')
        self.assertEqual(pack.parent, kokoro.VOICE_DIRECTORY)
        self.assertFalse(pack.exists())
        rows = {row['role']: row for row in kokoro.missing_artifacts('pf_dora')}
        self.assertIn('input_seal', rows)
        self.assertIn('model_config', rows)
        self.assertIn('model_weights', rows)
        self.assertIn('synthesis_runtime', rows)
        self.assertIn('voice_pack', rows)
        self.assertFalse(rows['synthesis_runtime']['installed'])
        self.assertFalse(rows['model_weights']['present'])
        self.assertEqual(rows['synthesis_runtime']['version'], '0.9.4')
        with self.assertRaises(ValueError):
            kokoro.missing_artifacts('nope')

    def test_kokoro_physical_entrypoints_are_held_and_explicitly_selectable(self):
        import kokoro_backend as kokoro
        with tempfile.TemporaryDirectory() as temp, patch('subprocess.Popen', side_effect=AssertionError('launch')):
            output = Path(temp)/'absent'
            with self.assertRaises(PlatformHeld):
                kokoro.KokoroBackend().run_stage('hello', output, 0)
            with self.assertRaises(PlatformHeld):
                kokoro.KokoroBackend('pf_dora').run_stage('olá', output, 0)
            self.assertFalse(output.exists())
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'job'
            with self.assertRaises(PlatformHeld):
                factory.main(['speak', '--speech-model', 'kokoro-82m',
                              '--speech-voice', 'pf_dora', '--out', str(output)])
            self.assertFalse(output.exists())
        # Text-model catalog identities remain separate from speech model choices.
        self.assertNotIn('kokoro', json.dumps(factory.MODELS).lower())
        for name in ('piper_backend.py', 'piper_worker.py'):
            with self.subTest(source=name):
                self.assertNotIn('kokoro', (factory.HERE/name).read_text().lower())
        self.assertEqual(factory.create_speech_backend().__class__.__name__, 'PiperBackend')
        selected = factory.create_speech_backend('kokoro-82m', 'pf_dora', environ={})
        self.assertIsInstance(selected, kokoro.KokoroBackend)
        self.assertEqual((selected.model_id, selected.voice_id), ('kokoro-82m', 'pf_dora'))
        with self.assertRaises(ValueError):
            factory.create_speech_backend('not-a-speech-model')
        with self.assertRaisesRegex(ValueError, 'only with --speech-model kokoro-82m'):
            factory.create_speech_backend('piper-cpu', 'pf_dora')
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); text = ContractBackend(); speech = SpeechFixture(text)
            result = factory._run_sequence('Prompt', ['qwen3-stock'], base/'job', base/'lease', text, speech)
            self.assertEqual(result['status'], 'ContractOnly')
            row = json.loads((base/'job/speech/result.json').read_text())
            self.assertEqual(row['sample_rate'], 22050)
            self.assertFalse((base/'job/speech/kokoro-result.json').exists())
        # A missing Kokoro asset is an explicit refusal, never a Piper fallback.
        with tempfile.TemporaryDirectory() as temp:
            stage = Path(temp)/'speech'; stage.mkdir()
            (stage/'request.json').write_text(json.dumps(
                {'model_id': 'kokoro-82m', 'text': 'hello', 'voice_id': 'af_heart',
                 'text_sha256': hashlib.sha256(b'hello').hexdigest()}))
            with self.assertRaises(kokoro.KokoroUnavailable):
                with patch('kokoro_backend.require_approved_kokoro_sample'):
                    kokoro.KokoroBackend().run_stage('hello', stage, time.monotonic() + 30)

    def test_kokoro_plan_selection_and_ptbr_pipeline_contract_are_explicit(self):
        import kokoro_backend as kokoro
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            prompt = base/'prompt.txt'; prompt.write_text('Olá, mundo!')
            output = base/'speech-plan.json'
            with patch('socket.socket', side_effect=AssertionError('socket attempted')), \
                 patch('subprocess.Popen', side_effect=AssertionError('process attempted')):
                self.assertEqual(factory.main(['speech-plan', '--prompt-file', str(prompt),
                                               '--models', 'bonsai-stock', '--out', str(output),
                                               '--speech-model', 'kokoro-82m',
                                               '--speech-voice', 'pf_dora']), 0)
            plan = json.loads(output.read_text())['speech']
            self.assertEqual(plan['model_id'], 'kokoro-82m')
            self.assertEqual(plan['voice_id'], 'pf_dora')
            self.assertEqual(plan['voices']['pf_dora']['language'], 'pt-BR')
            self.assertEqual(plan['sample_rate'], 24000)
            self.assertFalse(plan['readiness']['ready'])

        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); text = ContractBackend()
            speech = KokoroSpeechFixture(text, 'pf_dora')
            result = factory._run_sequence('Olá, mundo!', ['qwen3-stock'],
                                           base/'job', base/'lease', text, speech)
            stage = base/'job'/'speech'
            self.assertEqual(result['status'], 'ContractOnly')
            self.assertEqual(speech.request['model_id'], 'kokoro-82m')
            self.assertEqual(speech.request['voice_id'], 'pf_dora')
            self.assertEqual(speech.request['text'], 'contract text')
            receipt = json.loads((stage/'result.json').read_text())
            self.assertEqual(receipt['model_id'], 'kokoro-82m')
            self.assertEqual(receipt['voice_id'], 'pf_dora')
            self.assertEqual(receipt['voice_language'], 'pt-BR')
            self.assertEqual(receipt['sample_rate'], 24000)
            self.assertEqual(receipt['adapter'], 'kokoro-cpu')

    def test_kokoro_missing_assets_stop_before_text_or_output_without_fallback(self):
        import kokoro_backend as kokoro
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); text = ContractBackend()
            selected = factory.create_speech_backend('kokoro-82m', 'pf_dora', environ={})
            with self.assertRaises(kokoro.KokoroUnavailable):
                with patch('content_factory.require_real_execution'):
                    factory._run_sequence('Prompt', ['qwen3-stock'], base/'job',
                                          base/'lease', text, selected)
            self.assertEqual(text.events, [])
            self.assertFalse((base/'job').exists())

    def test_kokoro_worker_uses_local_pinned_api_and_emits_truthful_ptbr_and_english_wavs(self):
        from contextlib import nullcontext
        import kokoro_backend as kokoro
        import kokoro_worker as worker
        calls = {'models': [], 'pipelines': [], 'generators': []}

        class FakeTorch:
            def set_num_threads(self, count):
                calls['threads'] = count
            def set_num_interop_threads(self, count):
                calls['interop_threads'] = count
            def inference_mode(self):
                return nullcontext()

        class FakeModel:
            def __init__(self, **kwargs):
                calls['models'].append(kwargs)
            def to(self, device):
                calls['device'] = device
                return self
            def eval(self):
                calls['eval'] = True
                return self

        class FakeGenerator:
            def __init__(self):
                self.closed = False
            def __iter__(self):
                yield ('chunk one', 'phonemes', [0.0, 0.5, -1.0])
                yield ('chunk two', 'phonemes', [0.25])
            def close(self):
                self.closed = True

        class FakePipeline:
            def __init__(self, **kwargs):
                calls['pipelines'].append(kwargs)
            def __call__(self, text, **kwargs):
                calls['text'] = text
                calls['pipeline_call'] = kwargs
                generator = FakeGenerator()
                calls['generators'].append(generator)
                return generator

        class FakeRuntime:
            version = '0.9.4'
            torch = FakeTorch()
            KModel = FakeModel
            KPipeline = FakePipeline

        class FakeRuntimeLoader:
            actual_inference = False
            def __call__(self):
                return FakeRuntime

        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            model = base/'kokoro-v1_0.pth'; model.touch()
            config = base/'config.json'; config.touch()
            voices = base/'voices'; voices.mkdir()
            with patch.object(kokoro, 'MODEL_FILE', model), \
                 patch.object(kokoro, 'MODEL_CONFIG_FILE', config), \
                 patch.object(kokoro, 'VOICE_DIRECTORY', voices):
                for voice_id, text, lang_code, language in (
                        ('af_heart', 'Welcome home.', 'a', 'en-US'),
                        ('pf_dora', 'Bem-vinda ao estúdio.', 'p', 'pt-BR')):
                    stage = base/voice_id; stage.mkdir()
                    request = {'model_id': 'kokoro-82m', 'text': text,
                               'voice_id': voice_id,
                               'text_sha256': hashlib.sha256(text.encode()).hexdigest()}
                    receipt = worker.process_request(request, stage,
                                                     runtime_loader=FakeRuntimeLoader())
                    pipeline_args = calls['pipelines'][-1]
                    self.assertEqual(calls['models'][-1], {
                        'repo_id': 'hexgrad/Kokoro-82M',
                        'config': str(config), 'model': str(model)})
                    self.assertEqual(pipeline_args['lang_code'], lang_code)
                    self.assertEqual(pipeline_args['repo_id'], 'hexgrad/Kokoro-82M')
                    self.assertEqual(pipeline_args['device'], 'cpu')
                    self.assertIsInstance(pipeline_args['model'], FakeModel)
                    self.assertEqual(calls['device'], 'cpu')
                    self.assertEqual(calls['pipeline_call'], {
                        'voice': str(voices/(voice_id+'.pt')), 'speed': 1.0,
                        'split_pattern': r'\n+'})
                    self.assertTrue(calls['generators'][-1].closed)
                    self.assertEqual(calls['threads'], 4)
                    self.assertEqual(calls['interop_threads'], 1)
                    self.assertFalse(receipt['actual_inference'])
                    self.assertEqual(receipt['runtime_code_status'], 'mocked-runtime-boundary')
                    self.assertEqual(receipt['runtime_qualification'], 'unqualified')
                    self.assertEqual(receipt['runtime_version'], '0.9.4')
                    self.assertEqual(receipt['voice_language'], language)
                    self.assertEqual(receipt['sample_rate'], 24000)
                    with wave.open(str(stage/'answer.wav'), 'rb') as wav:
                        self.assertEqual((wav.getframerate(), wav.getnchannels(),
                                          wav.getsampwidth()), (24000, 1, 2))
                        frames = wav.readframes(wav.getnframes())
                    self.assertEqual(struct.unpack('<hhhh', frames), (0, 16384, -32768, 8192))

    def test_kokoro_worker_refuses_bad_audio_and_removes_invalid_saved_wav(self):
        from contextlib import nullcontext
        import kokoro_backend as kokoro
        import kokoro_worker as worker

        class FakeTorch:
            def set_num_threads(self, count): pass
            def set_num_interop_threads(self, count): pass
            def inference_mode(self): return nullcontext()

        class FakeModel:
            def __init__(self, **kwargs): pass
            def to(self, device): return self
            def eval(self): return self

        class FakeGenerator:
            def __init__(self, audio):
                self.audio = audio
                self.closed = False
            def __iter__(self):
                yield ('chunk', 'phonemes', self.audio)
            def close(self): self.closed = True

        class FakePipeline:
            audio = [0.0, 0.5]
            def __init__(self, **kwargs): pass
            def __call__(self, text, **kwargs):
                self.generator = FakeGenerator(self.audio)
                return self.generator

        class FakeRuntime:
            version = '0.9.4'
            torch = FakeTorch()
            KModel = FakeModel
            KPipeline = FakePipeline

        class FakeRuntimeLoader:
            actual_inference = False
            def __call__(self): return FakeRuntime

        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            worker._audio_values([float('nan')], kokoro.MAX_FRAMES)
        with self.assertRaisesRegex(ValueError, 'mono'):
            worker._audio_values([[0.1, 0.2], [0.3, 0.4]], kokoro.MAX_FRAMES)
        class Oversized:
            shape = (kokoro.MAX_FRAMES + 1,)
            def numel(self): return kokoro.MAX_FRAMES + 1
            def tolist(self): raise AssertionError('oversize tensor was materialized')
        with self.assertRaisesRegex(ValueError, '30-second'):
            worker._audio_values(Oversized(), kokoro.MAX_FRAMES)

        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); stage = base/'speech'; stage.mkdir()
            request = {'model_id': 'kokoro-82m', 'text': 'hello', 'voice_id': 'af_heart',
                       'text_sha256': hashlib.sha256(b'hello').hexdigest()}
            class EmptyAudioPipeline(FakePipeline):
                audio = None
            class EmptyRuntime(FakeRuntime):
                KPipeline = EmptyAudioPipeline
            class EmptyRuntimeLoader:
                actual_inference = False
                def __call__(self): return EmptyRuntime
            with self.assertRaisesRegex(ValueError, 'no audio'):
                worker.process_request(request, stage, runtime_loader=EmptyRuntimeLoader())
            self.assertFalse((stage/'answer.wav').exists())

            with patch.object(kokoro, 'validate_wav', side_effect=ValueError('bad saved WAV')):
                with self.assertRaisesRegex(ValueError, 'bad saved WAV'):
                    worker.process_request(request, stage, runtime_loader=FakeRuntimeLoader())
            self.assertFalse((stage/'answer.wav').exists())

    def test_kokoro_worker_rejects_runtime_mismatch_and_closes_generator_on_failure(self):
        from contextlib import nullcontext
        from types import SimpleNamespace
        import kokoro_worker as worker

        class FakeTorch:
            def set_num_threads(self, count): pass
            def set_num_interop_threads(self, count): pass
            def inference_mode(self): return nullcontext()

        calls = {'models': 0, 'generators': []}

        class FakeModel:
            def __init__(self, **kwargs):
                calls['models'] += 1
            def to(self, device): return self
            def eval(self): return self

        class FakeGenerator:
            closed = False
            def __iter__(self):
                raise RuntimeError('fake stream failure')
            def close(self): self.closed = True

        class FakePipeline:
            def __init__(self, **kwargs): pass
            def __call__(self, text, **kwargs):
                generator = FakeGenerator()
                calls['generators'].append(generator)
                return generator

        class FakeRuntime:
            version = '0.9.4'
            torch = FakeTorch()
            KModel = FakeModel
            KPipeline = FakePipeline

        class RuntimeMismatch:
            actual_inference = False
            def __call__(self):
                class WrongRuntime(FakeRuntime):
                    version = '0.9.3'
                return WrongRuntime

        class RuntimeLoader:
            actual_inference = False
            def __call__(self): return FakeRuntime

        with patch.object(worker, 'sys', SimpleNamespace(version_info=(3, 13, 0))), \
             patch('importlib.metadata.version', side_effect=AssertionError('package metadata read')):
            with self.assertRaisesRegex(RuntimeError, 'requires Python >=3.10,<3.13'):
                worker.load_runtime()

        request = {'model_id': 'kokoro-82m', 'text': 'hello', 'voice_id': 'af_heart',
                   'text_sha256': hashlib.sha256(b'hello').hexdigest()}
        with tempfile.TemporaryDirectory() as temp:
            stage = Path(temp)/'speech'; stage.mkdir()
            with self.assertRaisesRegex(RuntimeError, 'API version mismatch'):
                worker.process_request(request, stage, runtime_loader=RuntimeMismatch())
            self.assertFalse((stage/'answer.wav').exists())
            self.assertEqual(calls['models'], 0)

            with self.assertRaisesRegex(RuntimeError, 'fake stream failure'):
                worker.process_request(request, stage, runtime_loader=RuntimeLoader())
            self.assertTrue(calls['generators'][-1].closed)
            self.assertFalse((stage/'answer.wav').exists())

    def test_default_text_target_is_bonsai_without_qwen_substitution(self):
        self.assertEqual(factory.DEFAULT_MODEL, 'bonsai-stock')
        argv = factory.native_argv('bonsai-stock', 'Prompt')
        self.assertEqual(argv[0], factory.PRISM_NATIVE)
        self.assertTrue(argv[argv.index('--model')+1].endswith('Ternary-Bonsai-2-27B-PTQ1_0.gguf'))
        self.assertNotEqual(argv[0], factory.NATIVE)
        self.assertEqual(argv[argv.index('--reasoning')+1], 'off')

    def test_speech_command_fixed_and_supervised(self):
        argv = speech_guarded_argv(Path('/stage'), {'pid':123,'start_ticks':1,'pid_namespace':'pid:[1]'}, 10)
        self.assertIn('--max-vram-bytes', argv)
        self.assertIn('--parent-start-ticks', argv)
        self.assertEqual(argv[-2:], ['--request', '/stage/request.json'])
        self.assertTrue(argv[-3].endswith('piper_worker.py'))
        self.assertNotIn('/usr/bin/llama-completion', argv)
        for value in ('', '\0', 'é'*2049):
            with self.assertRaises(ValueError):
                check_speech_text(value)

    def test_wav_invalid_truncated_overduration_and_format_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'audio.wav'
            path.write_bytes(b'RIFF')
            with self.assertRaises((EOFError, wave.Error)):
                validate_wav(path)
            for rate, frames in ((16000, 20), (22050, 0), (22050, 22050*30+1)):
                with wave.open(str(path), 'wb') as wav:
                    wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(rate)
                    wav.writeframes(b'\0\0'*frames)
                with self.assertRaises(ValueError):
                    validate_wav(path)

    def test_shared_lease_busy_refuses_text_and_speech(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); text = ContractBackend(); speech = SpeechFixture(text)
            with (base/'lease').open('wb') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):
                    factory._run_sequence('Prompt', ['qwen3-stock'], base/'job', base/'lease', text, speech)
            self.assertEqual(text.events, [])
            self.assertFalse(speech.called)

    def test_insufficient_deadline_reserve_holds_speech(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); text = ContractBackend(); speech = SpeechFixture(text)
            with patch('content_factory.time.monotonic', side_effect=[0, 0, 0, 121]):
                with self.assertRaises(TimeoutError):
                    factory._run_sequence('Prompt', ['qwen3-stock'], base/'job', base/'lease', text, speech)
            self.assertFalse(speech.called)

    def test_bonsai_metadata_fit_binds_approved_prompt_and_template(self):
        spec = factory.MODELS['bonsai-stock']
        prompt = (factory.HERE/'first-run-prompt.txt').read_text()
        fit = fit_prompt(prompt, spec['path'], spec['template'], time.monotonic()+10, profile='bonsai-stock')
        self.assertEqual(fit['layers'], 65)
        self.assertEqual(fit['input_token_upper_bound'], 262)
        self.assertEqual(fit['output_tokens_reserved'], 64)
        self.assertEqual(fit['tensor_bytes_read'], 0)
        self.assertEqual(fit['context'], 512)
        self.assertLessEqual(fit['input_token_upper_bound']+64, 512)
        argv = factory.native_argv('bonsai-stock', prompt)
        self.assertEqual(argv[argv.index('--ctx-size')+1], '512')
        changed_argv = factory.native_argv('bonsai-stock', prompt+'!')
        self.assertEqual(changed_argv[changed_argv.index('--ctx-size')+1], '4096')
        require_prompt_fit({'prompt_fits':{fit['prompt_sha256']:fit}}, fit)
        with tempfile.TemporaryDirectory() as temp:
            changed = Path(temp)/'changed.jinja'
            changed.write_text('Wrong template')
            with self.assertRaises(ValueError):
                fit_prompt(prompt, spec['path'], changed, time.monotonic()+10, profile='bonsai-stock')

    def test_exact_prompt_fit_seal_required(self):
        fit = {'prompt_sha256':hashlib.sha256(b'a').hexdigest(), 'context':4096}
        seal = {'prompt_fits': {fit['prompt_sha256']:dict(fit)}}
        require_prompt_fit(seal, fit)
        for changed in ({**fit, 'context':2048}, {**fit, 'prompt_sha256':'different'}):
            with self.assertRaises(ValueError):
                require_prompt_fit(seal, changed)

    def test_bounded_gguf_header_rejects_corrupt_or_deadline(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'bad.gguf'; path.write_bytes(b'GGUF')
            with self.assertRaises(ValueError):
                gguf_metadata(path)
            with self.assertRaises(ValueError):
                gguf_metadata(path, time.monotonic()-1)

    def test_symlink_prompt_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); (base/'real').write_text('Hello'); (base/'link').symlink_to(base/'real')
            with self.assertRaises(OSError):
                factory.bounded_read(base/'link', 4096)


if __name__ == '__main__':
    unittest.main()
