import json
import re

from pydantic import BaseModel, Field


def _extract_json(content: str) -> dict:
    """
    Extract JSON from LLM response, handling markdown code blocks.
    """
    if not content:
        raise ValueError("Empty content provided")
    
    # Try to find JSON in markdown code blocks
    json_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    match = re.search(json_pattern, content, re.DOTALL)
    
    if match:
        json_str = match.group(1).strip()
    else:
        json_str = content.strip()
    
    # Try to parse JSON
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse JSON from LLM response: {str(e)}\n"
            f"Content: {content[:200]}"
        ) from e


def _normalize_recommendations_response(data: dict) -> dict:
    """
    Normalize LLM recommendations response to match RecommendationResult schema.
    """
    if "recommendations" not in data:
        raise ValueError("Recommendations response must contain 'recommendations' key")
    
    recommendations = data["recommendations"]
    if not isinstance(recommendations, list):
        recommendations = [recommendations]
    
    normalized_recs = []
    for rec in recommendations:
        if isinstance(rec, dict):
            # Convert priority to lowercase
            if "priority" in rec:
                rec["priority"] = rec["priority"].lower()
            
            # Ensure confidence is a float
            if "confidence" in rec:
                rec["confidence"] = float(rec["confidence"])
            
            normalized_recs.append(rec)
    
    data["recommendations"] = normalized_recs
    return data


class Recommendation(BaseModel):

    action: str

    reason: str

    priority: str

    expected_outcome: str

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )


class RecommendationResult(BaseModel):

    recommendations: list[Recommendation]


RECOMMENDATION_SYSTEM_PROMPT = """
You are an ML reliability engineer.

Generate concrete recommendations for investigating
or resolving a model regression.

Recommendations must be based only on the supplied
evidence.

Prioritize actions that can validate the suspected
root cause.

Do not recommend blindly retraining the model.

Return JSON matching the requested schema.
"""


def build_recommendation_prompt(
    evidence: dict,
    root_cause: dict,
) -> str:

    return f"""
REGRESSION EVIDENCE:

{json.dumps(evidence, indent=2)}

ROOT-CAUSE ANALYSIS:

{json.dumps(root_cause, indent=2)}

Generate prioritized recommendations based on the evidence and root-cause analysis.

Return a JSON object with EXACTLY this structure:

{{
  "recommendations": [
    {{
      "action": "concrete action to take",
      "reason": "why this action is needed",
      "priority": "high|medium|low",
      "expected_outcome": "what should happen if action is taken",
      "confidence": 0.0 to 1.0
    }},
    ...more recommendations...
  ]
}}

Requirements:
- priority MUST be one of: high, medium, low
- confidence MUST be a number between 0.0 and 1.0
- Rank recommendations by priority (high first)
- Base recommendations only on supplied evidence
- Do not recommend blindly retraining the model
- Do not invent metrics or observations
"""


class RecommendationEngine:

    def __init__(
        self,
        client,
        model: str,
    ) -> None:

        self.client = client
        self.model = model

    def generate(
        self,
        evidence: dict,
        root_cause: dict,
    ) -> RecommendationResult:

        response = (
            self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            RECOMMENDATION_SYSTEM_PROMPT
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            build_recommendation_prompt(
                                evidence,
                                root_cause,
                            )
                        ),
                    },
                ],
                temperature=0.1,
            )
        )

        content = (
            response.choices[0]
            .message
            .content
        )

        if not content:
            raise ValueError(
                "Empty recommendation response."
            )

        data = _extract_json(content)
        
        # Normalize the response to match schema
        data = _normalize_recommendations_response(data)

        return RecommendationResult.model_validate(
            data
        )