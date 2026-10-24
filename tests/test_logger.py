import csv
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from src.logger import QUERY_LOG_FIELDS, REQUEST_LOG_FIELDS, log_query, log_request


class LoggerTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = Path(
            self.stack.enter_context(tempfile.TemporaryDirectory(prefix="rag-log-test-"))
        ).resolve()
        self.assertEqual(self.directory.parent, Path(tempfile.gettempdir()).resolve())
        self.request_path = self.directory / "request_logs.csv"
        self.query_path = self.directory / "logs.csv"
        self.stack.enter_context(patch("src.logger.REQUEST_LOG_PATH", self.request_path))
        self.stack.enter_context(patch("src.logger.QUERY_LOG_PATH", self.query_path))
        self.stack.enter_context(patch("src.logger.ensure_project_dirs"))

    def test_success_and_failure_share_one_request_log_header(self):
        for request_id, status, error_type in (
            ("request-a", "success", ""),
            ("request-b", "failed", "TimeoutError"),
        ):
            log_request(request_id, "line 1,\nline 2", status, error_type, 12, 4)
        with self.request_path.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            self.assertEqual(reader.fieldnames, REQUEST_LOG_FIELDS)
            rows = list(reader)
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["status"] for row in rows], ["success", "failed"])
        self.assertEqual(rows[1]["error_type"], "TimeoutError")
        self.assertEqual(rows[1]["question"], "line 1,\nline 2")
        self.assertEqual(rows[1]["duration_ms"], "12")
        self.assertFalse(self.query_path.exists())

    def test_request_logging_keeps_legacy_query_file_unchanged(self):
        log_query("Q_A", "answer_A", [], 5, 4)
        before = self.query_path.read_bytes()
        log_request("request-b", "Q_B", "failed", "TimeoutError", 12, 4)
        self.assertEqual(self.query_path.read_bytes(), before)
        with self.query_path.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            self.assertEqual(reader.fieldnames, QUERY_LOG_FIELDS)
            rows = list(reader)
        self.assertEqual(rows[0]["model_answer"], "answer_A")


if __name__ == "__main__":
    unittest.main()
