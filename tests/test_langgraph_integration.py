"""Graph integration test for the semantic-audit agent.

The fake client models both the agent's two reasoning requests and its
additional structured grounding audit. No API key/network required.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

pytest.importorskip("langgraph")
from agent_core import CedarBridgeReAct


class FakeClient:
    def __init__(self):
        self.reasoning_calls = 0
        self.audit_calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        if "response_format" in kwargs:
            self.audit_calls += 1
            # The answer is grounded and no unanswered business question remains.
            kind = kwargs["response_format"]["json_schema"]["name"]
            if kind == "cedarbridge_source_coverage":
                data = {"resolution": "resolved", "logging_forbidden": False,
                        "reason": "The business documents explicitly exclude Chemistry."}
            else:
                data = {"source_answerability": "resolved", "logging_forbidden": False, "correction_needed": False, "needs_feedback": False,
                        "corrected_answer": "CedarBridge does not offer Chemistry.",
                        "reason": "Chemistry is excluded by the business sources."}
            msg = SimpleNamespace(content=json.dumps(data), tool_calls=[])
        else:
            self.reasoning_calls += 1
            if self.reasoning_calls == 1:
                call = SimpleNamespace(
                    id="call_1",
                    function=SimpleNamespace(
                        name="lookup_business_info",
                        arguments='{"question":"Chemistry"}'
                    )
                )
                msg = SimpleNamespace(content=None, tool_calls=[call])
            else:
                assert kwargs["messages"][-1]["role"] == "tool"
                msg = SimpleNamespace(
                    content="CedarBridge does not offer Chemistry.",
                    tool_calls=[]
                )
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_real_langgraph_nodes_edges_and_react_cycle():
    root = Path(__file__).resolve().parents[1]
    fake = FakeClient()
    agent = CedarBridgeReAct(client=fake, root=root, dry_run=True)
    result = agent.respond("Do you teach Chemistry?", persona="expert")
    assert "does not offer Chemistry" in result["answer"]
    assert result["reasoning_calls"] == 2
    assert result["verification_calls"] == 2
    assert result["model_calls"] == 4
    assert result["tool_calls"] == 1
    assert [x["step"] for x in result["trace"]] == [
        "reason", "act_observe", "reason", "grounding_audit"
    ]
    assert fake.reasoning_calls == 2
    assert fake.audit_calls == 2
