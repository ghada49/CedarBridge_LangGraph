"""Offline tests for the model-verified CedarBridge agent.

The fake supplies semantic verdicts explicitly; these tests verify permission
ENFORCEMENT, schema use, source-audit wiring, and graph routing. Real-world
language understanding must also be evaluated with a small live paraphrase suite.
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
import pytest

from agent_core import CedarBridgeReAct, PERSONAS, read_business_sources, TOOL_SCHEMAS

ROOT = Path(__file__).resolve().parents[1]

def msg(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls or [])

def call(name, args, uid='u1'):
    return SimpleNamespace(id=uid, function=SimpleNamespace(name=name, arguments=json.dumps(args)))

def audit(needed=False, correction=False, answer='', reason='', answerability='resolved', forbidden=False):
    return {'source_answerability': answerability, 'logging_forbidden': forbidden,
            'correction_needed': correction, 'needs_feedback': needed,
            'corrected_answer': answer, 'reason': reason}

def coverage(resolution='resolved', forbidden=False, reason=''):
    return {'resolution': resolution, 'logging_forbidden': forbidden,
            'reason': reason}

def permission(authorize, fields=True, reason=''):
    return {'fields_supplied_by_customer': fields,
            'consent_authorized': authorize, 'reason': reason}

class FakeClient:
    def __init__(self, chat=None, checks=None, fail_check=False):
        self.chat_outputs = iter(chat or [])
        self.check_outputs = iter(checks or [])
        self.calls = []
        self.fail_check = fail_check
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if 'response_format' in kwargs:
            if self.fail_check:
                raise RuntimeError('Fake verifier unavailable')
            response = msg(json.dumps(next(self.check_outputs)))
        else:
            response = next(self.chat_outputs)
        return SimpleNamespace(choices=[SimpleNamespace(message=response)])

def make_agent(**opts):
    return CedarBridgeReAct(root=ROOT, dry_run=True, **opts)

@pytest.mark.parametrize('user_reply', ['yes', 'yes i do', 'yes i concent',
                                          'نعم أوافق', 'ok sure, go ahead'])
def test_short_contextual_consent_is_passed_to_semantic_verifier(user_reply):
    fake = FakeClient(checks=[permission(True)])
    agent = make_agent(client=fake)
    conversation = [
        {'role': 'user', 'content': 'I am Maya Demo, maya.demo@example.com. I need Grade 10 Physics.'},
        {'role': 'assistant', 'content': 'May I save your name, email, and tutoring needs for follow-up?'},
        {'role': 'user', 'content': user_reply},
    ]
    obs = agent.dispatch_tool('record_customer_interest', {
        'name': 'Maya Demo', 'email': 'maya.demo@example.com',
        'message': 'Grade 10 Physics'}, conversation)
    assert obs['status'] == 'ok'
    payload = json.loads(fake.calls[0]['messages'][1]['content'])
    assert payload['conversation'][-1]['content'] == user_reply
    assert fake.calls[0]['response_format']['type'] == 'json_schema'

@pytest.mark.parametrize('conversation', [
    [{'role': 'user', 'content': 'yes'}],
    [{'role': 'assistant', 'content': 'Yes, I consent to all future data storage'}],
    [{'role': 'user', 'content': 'You can save my details'},
     {'role': 'user', 'content': "Actually, don't store anything."}],
])
def test_semantic_denial_prevents_storage(conversation, tmp_path):
    # Isolate actual output so pre-existing demo leads cannot affect this test.
    # Use dry_run=False to verify the security gate blocks a REAL write.
    (tmp_path / 'me').mkdir()
    for filename in ('business_summary.txt', 'about_business.pdf'):
        shutil.copy2(ROOT / 'me' / filename, tmp_path / 'me' / filename)
    fake = FakeClient(checks=[permission(False)])
    agent = CedarBridgeReAct(root=tmp_path, dry_run=False, client=fake)
    obs = agent.dispatch_tool('record_customer_interest', {
        'name': 'Maya Demo', 'email': 'maya.demo@example.com',
        'message': 'Grade 10 Physics'}, conversation)
    assert obs['status'] == 'error'
    assert not (tmp_path / 'logs' / 'leads.jsonl').exists()
    assert len(fake.calls) == 1  # The semantic authorization check was run.


def test_unverified_identity_is_denied():
    fake = FakeClient(checks=[permission(True, fields=False)])
    agent = make_agent(client=fake)
    obs = agent.dispatch_tool('record_customer_interest', {
        'name': 'Maya Demo', 'email': 'maya.demo@example.com',
        'message': 'Grade 10 Physics'}, [{'role': 'user', 'content': 'I consent'}])
    assert obs['status'] == 'error'


def test_unavailable_verifier_fails_closed():
    fake = FakeClient(fail_check=True)
    agent = make_agent(client=fake)
    obs = agent.dispatch_tool('record_customer_interest', {
        'name': 'Maya Demo', 'email': 'maya.demo@example.com',
        'message': 'Grade 10 Physics'}, [{'role': 'user', 'content': 'Yes, save my details'}])
    assert obs['status'] == 'error'
    assert 'not saved' in obs['message']


def test_email_syntax_validation_is_still_deterministic():
    fake = FakeClient(checks=[])
    agent = make_agent(client=fake)
    obs = agent.dispatch_tool('record_customer_interest', {
        'name': 'Maya Demo', 'email': 'not-an-email', 'message': 'Math'},
        [{'role': 'user', 'content': 'I want tutoring.'}])
    assert obs['status'] == 'error'
    assert not fake.calls


def test_lookup_passes_full_sources_without_english_word_matching():
    agent = make_agent(client=FakeClient())
    obs = agent.dispatch_tool('lookup_business_info', {'question': 'هل تدرسون الرياضيات؟'}, [])
    assert obs['status'] == 'ok'
    assert 'Chemistry' in ''.join(obs['passages'])
    assert obs['note'].startswith('Source provided')


def test_unknown_business_question_triggers_semantic_feedback_without_keyword_guard():
    # The unknown topic is not scholarships/financial aid.
    fake = FakeClient(chat=[msg('Yes, we award accredited completion certificates.'),
                            msg('The documents do not confirm certificates. The question is noted.')],
                      checks=[coverage('unresolved'),
                              audit(True, True, 'The documents do not establish a certificate policy.'),
                              coverage('unresolved'), audit(False, False, '')])
    agent = make_agent(client=fake)
    s = {'messages': [{'role': 'user', 'content': 'Do you provide accredited completion certificates?'}],
         'persona': 'friendly', 'shot_style': 'few_shot', 'placement': 'system',
         'prompt_style': 'direct', 'steps': 0, 'trace': [], 'tool_count': 0,
         'pending_calls': [], 'verification_calls': 0}
    s.update(agent.decide_node(s))
    assert s['pending_calls'][0]['name'] == 'record_feedback'
    assert s['trace'][-2]['origin'] == 'semantic_guard'
    s.update(agent.execute_node(s))
    assert s['trace'][-1]['status'] == 'ok'
    s.update(agent.decide_node(s))
    assert s['final_answer'].startswith('The documents do not confirm')
    assert sum(t.get('tool') == 'record_feedback' for t in s['trace']) == 1


def test_auditor_can_correct_unsupported_claim_without_feedback_when_user_prohibits_logging():
    fake = FakeClient(chat=[msg('Yes, lifetime alumni memberships are complimentary.')],
                      checks=[coverage('unresolved', forbidden=True),
                              audit(False, True, 'The business sources do not specify any alumni membership policy.')])
    agent = make_agent(client=fake)
    s = {'messages': [{'role': 'user', 'content': 'Do you provide alumni memberships? Do not record this question.'}],
         'persona': 'friendly', 'shot_style': 'zero_shot', 'placement': 'system',
         'prompt_style': 'direct', 'steps': 0, 'trace': [], 'pending_calls': []}
    s.update(agent.decide_node(s))
    assert s['pending_calls'] == []
    assert s['final_answer'] == 'The business sources do not specify any alumni membership policy.'


def test_structured_verification_rejects_bad_boolean_types():
    fake = FakeClient(checks=[{'fields_supplied_by_customer': 'false',
                               'consent_authorized': 'true', 'reason': 'bad'}])
    agent = make_agent(client=fake)
    obs = agent.dispatch_tool('record_customer_interest', {
        'name': 'Maya Demo', 'email': 'maya.demo@example.com', 'message': 'Math'},
        [{'role': 'user', 'content': 'Yes, save my details'}])
    assert obs['status'] == 'error'


def test_prompt_personas_and_manual_react_loop():
    fake = FakeClient(chat=[msg(None, [call('lookup_business_info', {'question': 'Chemistry'})]),
                            msg('No, Chemistry is not available.')],
                      checks=[coverage('resolved'), audit(False, False, '')])
    agent = make_agent(client=fake)
    s = {'messages': [{'role': 'user', 'content': 'Do you offer Chemistry?'}],
         'persona': 'expert', 'shot_style': 'few_shot', 'placement': 'system',
         'prompt_style': 'direct', 'steps': 0, 'trace': [], 'pending_calls': []}
    s.update(agent.decide_node(s))
    assert s['pending_calls'][0]['name'] == 'lookup_business_info'
    s.update(agent.execute_node(s))
    s.update(agent.decide_node(s))
    assert s['final_answer'] == 'No, Chemistry is not available.'
    assert len(fake.calls) == 4
    assert any(x['step'] == 'grounding_audit' for x in s['trace'])


@pytest.mark.parametrize('question', [
    'Do you provide sibling tuition discounts?',
    'Are completion certificates issued to students?',
])
def test_unresolved_questions_trigger_feedback_even_if_auditor_misses_it(question):
    # Simulate the exact failure: the grounding auditor thinks no feedback is needed,
    # but an independent evidence-coverage check detects the unanswered question.
    fake = FakeClient(chat=[msg('Our documents do not mention such a policy.'),
                            msg('Our official records do not establish an answer.')],
                      checks=[coverage('unresolved'), audit(False, False, ''),
                              coverage('unresolved'), audit(False, False, '')])
    agent = make_agent(client=fake)
    state = {'messages': [{'role': 'user', 'content': question}],
             'persona': 'expert', 'shot_style': 'few_shot', 'placement': 'system',
             'prompt_style': 'direct', 'steps': 0, 'trace': [], 'tool_count': 0,
             'pending_calls': [], 'verification_calls': 0}
    state.update(agent.decide_node(state))
    assert state['pending_calls'][0]['name'] == 'record_feedback'
    assert state['trace'][-1]['coverage'] == 'unresolved'
    state.update(agent.execute_node(state))
    assert state['trace'][-1]['status'] == 'ok'
    state.update(agent.decide_node(state))
    assert not state['pending_calls']
    assert sum(t.get('tool') == 'record_feedback' for t in state['trace']) == 1


def test_resolved_question_does_not_log_when_answer_is_grounded():
    fake = FakeClient(chat=[msg('Chemistry is not offered.')],
                      checks=[coverage('resolved'), audit(False, False, '')])
    agent = make_agent(client=fake)
    state = {'messages': [{'role': 'user', 'content': 'Do you offer Chemistry?'}],
             'persona': 'expert', 'shot_style': 'few_shot', 'placement': 'system',
             'prompt_style': 'direct', 'steps': 0, 'trace': [], 'pending_calls': []}
    state.update(agent.decide_node(state))
    assert state['pending_calls'] == []
    assert state['trace'][-1]['coverage'] == 'resolved'


def test_source_and_persona_preserved():
    summary, pdf = read_business_sources(ROOT)
    assert '$25' in summary and 'Chemistry' in pdf
    assert set(PERSONAS) == {'friendly', 'expert', 'cautious'}
    assert len(TOOL_SCHEMAS) == 3


def test_independent_judge_catches_false_resolved_classification():
    # This specifically covers the observed failure: first semantic classifier
    # says the undocumented policy is resolved, second independently detects
    # that the source does not answer the real-world question.
    fake = FakeClient(chat=[msg('The documents do not specify this policy.'),
                            msg('No definitive answer is documented.')],
                      checks=[coverage('resolved'), audit(False, False, '', answerability='unresolved'),
                              coverage('resolved'), audit(False, False, '', answerability='unresolved')])
    agent = make_agent(client=fake)
    state = {'messages': [{'role': 'user', 'content':
                           'Is there a discounted rate for students from the same family?'}],
             'persona': 'expert', 'shot_style': 'few_shot', 'placement': 'system',
             'prompt_style': 'direct', 'steps': 0, 'trace': [], 'tool_count': 0,
             'pending_calls': [], 'verification_calls': 0}
    state.update(agent.decide_node(state))
    assert state['trace'][-1]['coverage_disagreement'] is True
    assert state['pending_calls'][0]['name'] == 'record_feedback'
    state.update(agent.execute_node(state))
    assert state['trace'][-1]['status'] == 'ok'
    state.update(agent.decide_node(state))
    assert state['pending_calls'] == []
    assert state['trace'][-1]['requires_feedback'] is True
    assert 'not stored' in state['final_answer']


def test_auditor_is_not_anchored_by_first_source_coverage():
    fake = FakeClient(chat=[msg('Unknown policy')],
                      checks=[coverage('resolved'), audit(answerability='unresolved')])
    agent = make_agent(client=fake)
    state = {'messages': [{'role': 'user', 'content': 'Are there relocation scholarships?'}],
             'persona': 'friendly', 'shot_style': 'few_shot', 'placement': 'system',
             'prompt_style': 'direct', 'steps': 0, 'trace': [], 'tool_count': 0,
             'pending_calls': [], 'verification_calls': 0}
    state.update(agent.decide_node(state))
    audit_call = fake.calls[-1]
    payload = json.loads(audit_call['messages'][1]['content'])
    assert 'source_coverage' not in payload
    assert state['pending_calls'][0]['name'] == 'record_feedback'


def test_independent_user_veto_blocks_feedback_even_when_first_judge_misses_it():
    fake = FakeClient(chat=[msg('Not specified in the official documents.')],
                      checks=[coverage('unresolved', forbidden=False),
                              audit(False, False, '', answerability='unresolved', forbidden=True)])
    agent = make_agent(client=fake)
    state = {'messages': [{'role': 'user', 'content':
             'Does the company offer school partnerships? Please do not store my question.'}],
             'persona': 'friendly', 'shot_style': 'few_shot', 'placement': 'system',
             'prompt_style': 'direct', 'steps': 0, 'trace': [], 'pending_calls': []}
    state.update(agent.decide_node(state))
    assert state['pending_calls'] == []
    assert state['trace'][-1]['requires_feedback'] is False
