import contextvars
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama
from ollama import Client

import api
import files
import performance
import rag
from model_runtime import configure_cpu_threads, keep_alive_seconds
from performance import invoke_chat, request_profile, timed_stage
from request_queue import QuestionJob


class RuntimeTests(unittest.TestCase):
    def test_actual_chat_photo_and_embedding_payloads_keep_models_loaded(self):
        requests = []

        def respond(request):
            body = json.loads(request.content)
            requests.append(body)
            if request.url.path == "/api/embed":
                return httpx.Response(200, json={"embeddings": [[0.1, 0.2]]})
            return httpx.Response(200, json={
                "model": body["model"], "done": True,
                "message": {"role": "assistant", "content": "Ответ"},
            })

        client = Client(transport=httpx.MockTransport(respond))
        # Exercise the actual app's vision and embedding clients and the same
        # chat adapter used by the graph, without contacting Ollama or a GPU.
        chat = ChatOllama(model="test-chat", keep_alive=keep_alive_seconds(), reasoning=False)
        with patch.object(chat, "_client", client), \
             patch.object(files.vision_llm, "_client", client), \
             patch.object(rag.embeddings, "_client", client):
            self.assertEqual(chat.invoke([HumanMessage(content="Вопрос")]).content, "Ответ")
            self.assertEqual(files.describe_photo(b"test-image"), "Ответ")
            self.assertEqual(rag.embeddings.embed_query("Почва"), [0.1, 0.2])
        self.assertEqual(len(requests), 3)
        self.assertTrue(all(body["keep_alive"] == 1800 for body in requests))

    def test_retention_can_be_changed_or_disabled(self):
        for value, expected in [("600", 600), ("0", 0), ("-1", -1)]:
            with self.subTest(value=value), patch.dict(os.environ, {"OLLAMA_KEEP_ALIVE_SECONDS": value}):
                self.assertEqual(keep_alive_seconds(), expected)
        with patch.dict(os.environ, {"OLLAMA_KEEP_ALIVE_SECONDS": "-2"}), self.assertRaises(ValueError):
            keep_alive_seconds()

    def test_cpu_threads_use_hardware_defaults_and_allow_explicit_tuning(self):
        torch = Mock()
        with patch.dict(os.environ, {"RAG_CPU_THREADS": ""}):
            configure_cpu_threads(torch)
        torch.set_num_threads.assert_not_called()
        with patch.dict(os.environ, {"RAG_CPU_THREADS": "12"}):
            configure_cpu_threads(torch)
        torch.set_num_threads.assert_called_once_with(12)
        with patch.dict(os.environ, {"RAG_CPU_THREADS": "0"}), self.assertRaises(ValueError):
            configure_cpu_threads(torch)


class PerformanceTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {"PERFORMANCE_LOGGING": "1"})
        environment.start()
        self.addCleanup(environment.stop)

    def test_queue_and_processing_are_separate_and_metadata_is_private(self):
        message = AIMessage(content="PRIVATE ANSWER", response_metadata={
            "load_duration": 24_000_000_000,
            "prompt_eval_duration": 100_000_000,
            "eval_duration": 64_000_000,
            "prompt_eval_count": 21, "eval_count": 3,
            "message": "PRIVATE METADATA", "token": "PRIVATE TOKEN",
        })
        model = SimpleNamespace(invoke=Mock(return_value=message))
        messages = [HumanMessage(content="PRIVATE QUESTION")]
        with patch.object(performance.time, "monotonic", side_effect=[10, 11, 36, 40]), \
             self.assertLogs(performance.logger, level="INFO") as logs:
            with request_profile("abcdefgh-full-private-job-token", submitted_at=5):
                self.assertIs(invoke_chat(model, messages, "answer"), message)
        model.invoke.assert_called_once_with(messages)
        text = "\n".join(logs.output)
        for expected in ["[timing abcdefgh]", "answer=25.000s", "load_s=24.000",
                         "prompt_s=0.100", "generate_s=0.064", "input_tokens=21",
                         "queue=5.000s processing=30.000s total=35.000s status=ok"]:
            self.assertIn(expected, text)
        self.assertNotIn("PRIVATE", text)
        self.assertNotIn("full-private-job-token", text)

    def test_failed_requests_emit_timings_and_clear_context(self):
        with self.assertLogs(performance.logger, level="INFO") as logs:
            with self.assertRaisesRegex(RuntimeError, "PRIVATE FAILURE"):
                with request_profile("error-id", submitted_at=0):
                    with timed_stage("answer"):
                        raise RuntimeError("PRIVATE FAILURE")
        self.assertIn("status=failed", "\n".join(logs.output))
        self.assertNotIn("PRIVATE FAILURE", "\n".join(logs.output))
        with self.assertNoLogs(performance.logger, level="INFO"):
            with timed_stage("unrelated") as details:
                self.assertIsNone(details)

    def test_disabled_profiles_and_direct_calls_do_not_log(self):
        model = SimpleNamespace(invoke=Mock(return_value=AIMessage(content="Ответ")))
        with patch.dict(os.environ, {"PERFORMANCE_LOGGING": "0"}), \
             self.assertNoLogs(performance.logger, level="INFO"):
            with request_profile("disabled", submitted_at=0):
                invoke_chat(model, [], "answer")
        with self.assertNoLogs(performance.logger, level="INFO"):
            invoke_chat(model, [], "direct")

    def test_untrusted_metadata_cannot_break_response_or_enter_logs(self):
        for metadata in [None, {"load_duration": "PRIVATE", "eval_duration": float("nan"),
                                "prompt_eval_count": True, "eval_count": -3}]:
            model = SimpleNamespace(invoke=Mock(return_value=SimpleNamespace(
                content="Ответ", response_metadata=metadata)))
            with self.assertLogs(performance.logger, level="INFO") as logs:
                with request_profile("metadata", submitted_at=0):
                    self.assertEqual(invoke_chat(model, [], "answer").content, "Ответ")
            text = "\n".join(logs.output)
            self.assertNotIn("PRIVATE", text)
            self.assertNotIn("load_s", text)
            self.assertNotIn("generate_s", text)
            self.assertNotIn("output_tokens", text)

    def test_independent_contexts_do_not_mix_stages(self):
        other = contextvars.Context()
        with self.assertLogs(performance.logger, level="INFO") as logs:
            with request_profile("first-id", submitted_at=0):
                def independent():
                    with request_profile("second-id", submitted_at=0):
                        with timed_stage("photo"):
                            pass
                other.run(independent)
                with timed_stage("answer"):
                    pass
        photo = [line for line in logs.output if "photo=" in line]
        answer = [line for line in logs.output if "answer=" in line]
        self.assertEqual(len(photo), 1)
        self.assertEqual(len(answer), 1)
        self.assertIn("[timing second-i]", photo[0])
        self.assertIn("[timing first-id]", answer[0])

    def test_api_records_real_job_wait_without_changing_answer_or_history(self):
        job = QuestionJob(text="PRIVATE QUESTION", image=None, user_id=1, submitted_at=2)
        with patch.object(api, "run_agent", return_value={"answer": "Ответ", "intent": "science"}), \
             patch.object(api.db, "save_conversation_turn", return_value=True) as save, \
             patch.object(performance.time, "monotonic", side_effect=[10, 10, 14, 14]), \
             self.assertLogs(performance.logger, level="INFO") as logs:
            self.assertEqual(api.process_question(job), "Ответ")
        save.assert_called_once_with(None, 1, "PRIVATE QUESTION", "Ответ", "science", 4, False, None)
        text = "\n".join(logs.output)
        self.assertIn("queue=8.000s processing=4.000s total=12.000s", text)
        self.assertNotIn("PRIVATE QUESTION", text)


if __name__ == "__main__":
    unittest.main()
