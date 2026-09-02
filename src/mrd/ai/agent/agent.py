import json

from openai import OpenAI


AGENT_SYSTEM_PROMPT = """
You are an ML reliability engineering agent.

You investigate model regressions.

You have access to tools that can retrieve:

- historical incidents
- engineering documentation
- feature documentation
- ML knowledge

You must investigate before reaching conclusions.

Rules:

1. Never invent evidence.
2. Clearly distinguish facts from hypotheses.
3. Use tools when additional information is useful.
4. Prefer multiple independent pieces of evidence.
5. If evidence is insufficient, say so.
6. Never claim a hypothesis is proven unless evidence supports it.
"""


class RegressionAgent:

    def __init__(
        self,
        client: OpenAI,
        model: str,
        tools,
    ) -> None:

        self.client = client
        self.model = model
        self.tools = tools

    def investigate(
        self,
        evidence: dict,
    ) -> str:

        messages = [
            {
                "role": "system",
                "content": AGENT_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    "Investigate this regression:\n\n"
                    + json.dumps(
                        evidence,
                        indent=2,
                    )
                ),
            },
        ]

        for _ in range(8):

            response = (
                self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools.get_tool_definitions(),
                    tool_choice="auto",
                    temperature=0.1,
                )
            )

            message = response.choices[0].message

            if not message.tool_calls:

                return message.content or ""

            messages.append(
                message.model_dump()
            )

            for tool_call in message.tool_calls:

                arguments = json.loads(
                    tool_call.function.arguments
                )

                result = self.tools.execute(
                    name=tool_call.function.name,
                    arguments=arguments,
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": (
                            tool_call.id
                        ),
                        "content": json.dumps(
                            result,
                            default=str,
                        ),
                    }
                )

        raise RuntimeError(
            "AI investigation exceeded maximum steps."
        )