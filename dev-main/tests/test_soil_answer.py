"""Transport and evidence tests, not a substitute for live recognition accuracy."""
import base64
import json
import os
import unittest
from unittest.mock import Mock, patch

import httpx
from langchain_core.messages import AIMessage, HumanMessage
from ollama import Client

import soil_answer


class SoilAnswerTests(unittest.TestCase):
    def test_real_adapter_sends_thinking_pixels_history_and_returns_only_final_answer(self):
        requests = []

        def respond(request):
            body = json.loads(request.content)
            requests.append(body)
            return httpx.Response(200, json={
                "model": body["model"], "done": True,
                "message": {"role": "assistant", "content": "Предварительно — подзол.",
                            "thinking": "PRIVATE REASONING"},
            })

        client = Client(transport=httpx.MockTransport(respond))
        with patch.object(soil_answer.text_llm, "_client", client), \
             patch.object(soil_answer.vision_llm, "_client", client):
            for image in (None, b"normalized-jpeg"):
                with self.subTest(image=bool(image)):
                    answer = soil_answer.answer(
                        "Ленинградская область, сосновый лес, анализов нет",
                        image=image, user_notes=["Светлый слой, ниже рыжий"],
                        history=[HumanMessage(content="Что за почва?"), AIMessage(content="Где участок?")])
                    self.assertEqual(answer, "Предварительно — подзол.")
                    body = requests[-1]
                    self.assertIs(body["think"], True)
                    self.assertNotEqual(body.get("format"), "json")
                    self.assertEqual(body["options"]["num_predict"], 8192)
                    self.assertEqual(body["keep_alive"], 1800)
                    self.assertEqual([row["role"] for row in body["messages"]],
                                     ["system", "user", "assistant", "user"])
                    message = body["messages"][-1]
                    self.assertIn("Светлый слой, ниже рыжий", message["content"])
                    if image:
                        self.assertEqual(base64.b64decode(message["images"][0]), image)
                    else:
                        self.assertFalse(message.get("images"))

    def test_reference_is_auxiliary_and_legacy_gates_are_not_sent(self):
        context = soil_answer.reference_context()
        self.assertIn("Подзолы:", context)
        self.assertIn("https://soils.narod.ru/taxon/type/podz07.html", context)
        for legacy in ("mandatory_features", "minimum_features", "не являются основанием для гипотезы"):
            self.assertNotIn(legacy, context)
        self.assertIn("Отсутствие анализов не запрещает", soil_answer.PROMPT)

    def test_unavailable_reference_does_not_prevent_a_model_answer(self):
        with patch.object(soil_answer, "reference", side_effect=OSError), \
             patch.object(soil_answer, "text_llm", Mock(invoke=Mock(return_value=AIMessage(content="Возможен подзол.")))) as model:
            self.assertEqual(soil_answer.answer("Что за почва?"), "Возможен подзол.")
            model.invoke.assert_called_once()

    def test_hidden_thinking_never_becomes_a_fallback_answer(self):
        for content, expected in (("<think>PRIVATE</think>Ответ", "Ответ"),
                                  ("", None), ("<think>PRIVATE", None)):
            model = Mock()
            model.invoke.return_value = AIMessage(
                content=content, additional_kwargs={"reasoning_content": "PRIVATE"})
            with self.subTest(content=content), patch.object(soil_answer, "text_llm", model):
                if expected:
                    self.assertEqual(soil_answer.answer("Вопрос"), expected)
                else:
                    with self.assertRaises(RuntimeError):
                        soil_answer.answer("Вопрос")

    def test_grounding_policy_changes_only_the_system_instruction(self):
        kwargs = dict(image=b"same-photo", user_notes=["На фото вода; обычно не знаю"],
                      history=[HumanMessage(content="Что это?"), AIMessage(content="Глеезём.")])
        baseline = soil_answer.build_messages("Только фото", **kwargs)
        grounded = soil_answer.build_messages("Только фото", policy="grounded", **kwargs)
        self.assertEqual(baseline[0].content, soil_answer.PROMPT)
        self.assertNotEqual(baseline[0].content, grounded[0].content)
        self.assertEqual(baseline[1:], grounded[1:])
        self.assertIn(soil_answer.GROUNDING_POLICY, grounded[0].content)

    def test_policy_is_opt_in_and_invalid_policy_never_calls_model(self):
        model = Mock(invoke=Mock(return_value=AIMessage(content="Ответ.")))
        with patch.object(soil_answer, "text_llm", model), \
             patch.dict(os.environ, {"SOIL_ANSWER_POLICY": "grounded"}):
            self.assertEqual(soil_answer.answer("Что за почва?"), "Ответ.")
            self.assertEqual(model.invoke.call_args.args[0][0].content,
                             soil_answer.SYSTEM_PROMPTS["grounded"])
            model.reset_mock()
            with patch.dict(os.environ, {"SOIL_ANSWER_POLICY": "typo"}):
                with self.assertRaises(ValueError):
                    soil_answer.answer("Что за почва?")
            model.invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
