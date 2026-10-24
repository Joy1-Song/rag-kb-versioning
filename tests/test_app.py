import os
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def answer_payload(answer):
    return {
        "answer": answer,
        "sources": ["test.txt"],
        "retrieved_docs": [
            {"chunk_id": "test-1", "source": "test.txt", "content": "Test evidence."}
        ],
        "latency_ms": 1,
    }


class AppRequestTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(
            patch.dict(os.environ, {"DASHSCOPE_API_KEY": "offline-test-placeholder"})
        )
        self.stack.enter_context(
            patch(
                "socket.socket.connect",
                side_effect=AssertionError("Network access is disabled in these tests."),
            )
        )
        for name, value in (
            ("DEFAULT_CHUNK_SIZE", 600),
            ("DEFAULT_CHUNK_OVERLAP", 100),
            ("DEFAULT_TOP_K", 4),
        ):
            self.stack.enter_context(patch("src.config." + name, value))

        self.service = MagicMock()
        self.factory = self.stack.enter_context(
            patch("src.rag_service.RagService", return_value=self.service)
        )
        self.request_log = self.stack.enter_context(patch("src.logger.log_request"))
        self.query_log = self.stack.enter_context(patch("src.logger.log_query"))
        self.eval_log = self.stack.enter_context(patch("src.logger.log_eval"))
        self.app = AppTest.from_file(str(APP_PATH), default_timeout=15).run()
        self.assertFalse(self.app.exception)

    def submit(self, question):
        self.app.chat_input[0].set_value(question).run()
        self.assertFalse(self.app.exception)

    def test_success_then_timeout_clears_old_answer_and_blocks_evaluation(self):
        self.service.ask.side_effect = [answer_payload("answer_A"), TimeoutError("test timeout")]
        self.submit("Q_A")
        self.assertEqual(self.app.session_state["last_answer_payload"]["answer"], "answer_A")

        self.submit("Q_B")
        self.assertEqual(self.app.session_state["last_question"], "Q_B")
        self.assertIsNone(self.app.session_state["last_answer_payload"])
        self.assertEqual(
            [message["role"] for message in self.app.session_state["messages"]],
            ["user", "assistant", "user"],
        )
        self.assertEqual(self.app.session_state["messages"][1]["content"], "answer_A")
        self.app.button(key="evaluate_answer").click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(len(self.app.error), 1)
        self.eval_log.assert_not_called()
        self.query_log.assert_called_once()

        rows = [call.kwargs for call in self.request_log.call_args_list]
        self.assertEqual([row["status"] for row in rows], ["success", "failed"])
        self.assertEqual(rows[1]["question"], "Q_B")
        self.assertEqual(rows[1]["error_type"], "TimeoutError")
        self.assertEqual(rows[0]["error_type"], "")
        self.assertNotEqual(rows[0]["request_id"], rows[1]["request_id"])
        self.assertTrue(all(row["duration_ms"] >= 0 for row in rows))

    def test_first_request_timeout_is_logged(self):
        self.service.ask.side_effect = TimeoutError("test timeout")
        self.submit("Q_B")
        self.assertIsNone(self.app.session_state["last_answer_payload"])
        self.request_log.assert_called_once()
        self.assertEqual(self.request_log.call_args.kwargs["status"], "failed")
        self.assertEqual(self.request_log.call_args.kwargs["error_type"], "TimeoutError")
        self.query_log.assert_not_called()

    def test_success_after_timeout_can_be_saved_and_evaluated(self):
        self.service.ask.side_effect = [
            answer_payload("answer_A"), TimeoutError("test timeout"), answer_payload("answer_C")
        ]
        for question in ("Q_A", "Q_B", "Q_C"):
            self.submit(question)
        self.assertEqual(self.app.session_state["last_answer_payload"]["answer"], "answer_C")
        self.app.button(key="evaluate_answer").click().run()
        self.assertFalse(self.app.exception)
        self.eval_log.assert_called_once()
        self.assertEqual(self.eval_log.call_args.kwargs["question"], "Q_C")
        self.assertEqual(self.eval_log.call_args.kwargs["model_answer"], "answer_C")
        self.assertEqual(self.request_log.call_count, 3)
        self.assertEqual(self.query_log.call_count, 2)
        self.factory.assert_called_once()

    def test_initialization_failure_is_caught_and_logged(self):
        self.factory.side_effect = RuntimeError("offline initialization failure")
        self.submit("Q_A")
        self.assertIsNone(self.app.session_state["last_answer_payload"])
        self.assertNotIn("rag_service", self.app.session_state)
        self.assertEqual(len(self.app.error), 1)
        self.assertEqual(self.request_log.call_args.kwargs["error_type"], "RuntimeError")
        self.assertEqual(self.request_log.call_args.kwargs["status"], "failed")
        self.service.ask.assert_not_called()

    def test_missing_key_clears_previous_answer_without_calling_service(self):
        self.service.ask.return_value = answer_payload("answer_A")
        self.submit("Q_A")
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": ""}):
            self.submit("Q_B")
        self.assertIsNone(self.app.session_state["last_answer_payload"])
        self.service.ask.assert_called_once()
        self.assertEqual(self.request_log.call_args.kwargs["error_type"], "MissingAPIKey")
        self.assertEqual(self.request_log.call_args.kwargs["status"], "failed")

    def test_request_log_failure_does_not_discard_successful_answer(self):
        self.service.ask.return_value = answer_payload("answer_A")
        self.request_log.side_effect = OSError("test disk failure")
        self.submit("Q_A")
        self.assertEqual(self.app.session_state["last_answer_payload"]["answer"], "answer_A")
        self.assertEqual(len(self.app.warning), 1)
        self.query_log.assert_called_once()

    def test_request_log_failure_does_not_hide_timeout(self):
        self.service.ask.side_effect = TimeoutError("test timeout")
        self.request_log.side_effect = OSError("test disk failure")
        self.submit("Q_B")
        self.assertIsNone(self.app.session_state["last_answer_payload"])
        self.assertEqual(len(self.app.error), 1)
        self.assertIn("test timeout", self.app.error[0].value)
        self.assertEqual(len(self.app.warning), 1)

    def test_detail_log_failure_does_not_discard_successful_answer(self):
        self.service.ask.return_value = answer_payload("answer_A")
        self.query_log.side_effect = OSError("test disk failure")
        self.submit("Q_A")
        self.assertEqual(self.app.session_state["last_answer_payload"]["answer"], "answer_A")
        self.assertEqual(len(self.app.warning), 1)
        self.assertEqual(self.request_log.call_args.kwargs["status"], "success")


if __name__ == "__main__":
    unittest.main()
