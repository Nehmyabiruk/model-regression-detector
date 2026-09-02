from typing import Any


class InvestigationTools:

    def __init__(
        self,
        memory,
        rag,
    ) -> None:

        self.memory = memory
        self.rag = rag

    def search_history(
        self,
        model_name: str,
    ) -> list[dict]:

        incidents = self.memory.get_by_model(
            model_name
        )

        return [
            incident.model_dump()
            for incident in incidents
        ]

    def search_knowledge(
        self,
        query: str,
    ) -> str:

        return self.rag.build_context(
            query=query,
            limit=5,
        )

    def get_tool_definitions(self) -> list[dict]:

        return [
            {
                "type": "function",
                "function": {
                    "name": "search_history",
                    "description": (
                        "Search previous regression "
                        "incidents for a model."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "model_name": {
                                "type": "string"
                            }
                        },
                        "required": [
                            "model_name"
                        ],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_knowledge",
                    "description": (
                        "Search ML documentation, "
                        "feature documentation, "
                        "and engineering knowledge."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string"
                            }
                        },
                        "required": [
                            "query"
                        ],
                    },
                },
            },
        ]

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> Any:

        if name == "search_history":
            return self.search_history(
                **arguments
            )

        if name == "search_knowledge":
            return self.search_knowledge(
                **arguments
            )

        raise ValueError(
            f"Unknown tool: {name}"
        )