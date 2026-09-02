from __future__ import annotations
import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

from mrd.ai.prompts import (
    SYSTEM_PROMPT,
    build_investigation_prompt,
)
from mrd.ai.schemas import AIInvestigation

from mrd.ai.prompts import (
    RCA_SYSTEM_PROMPT,
    build_rca_prompt,
)

from mrd.ai.rca import RootCauseAnalysis
load_dotenv()


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


def _normalize_investigation_response(data: dict) -> dict:
    """
    Normalize LLM response to match AIInvestigation schema.
    Handles common variations in LLM output formatting.
    """
    # Convert severity to lowercase
    if "severity" in data:
        data["severity"] = data["severity"].lower()
    
    # Convert evidence to list if it's a string
    if "evidence" in data and isinstance(data["evidence"], str):
        # Split by newlines or bullet points
        evidence_text = data["evidence"]
        # Try to split by common delimiters
        if "\n" in evidence_text:
            data["evidence"] = [item.strip() for item in evidence_text.split("\n") if item.strip()]
        elif ";" in evidence_text:
            data["evidence"] = [item.strip() for item in evidence_text.split(";") if item.strip()]
        else:
            # If no delimiters, wrap as single-item list
            data["evidence"] = [evidence_text.strip()]
    
    # Convert affected_areas to list if it's a string
    if "affected_areas" in data and isinstance(data["affected_areas"], str):
        areas_text = data["affected_areas"]
        if "\n" in areas_text:
            data["affected_areas"] = [item.strip() for item in areas_text.split("\n") if item.strip()]
        elif ";" in areas_text:
            data["affected_areas"] = [item.strip() for item in areas_text.split(";") if item.strip()]
        else:
            data["affected_areas"] = [areas_text.strip()]
    
    # Convert recommendations to list if it's a string
    if "recommendations" in data and isinstance(data["recommendations"], str):
        rec_text = data["recommendations"]
        if "\n" in rec_text:
            data["recommendations"] = [item.strip() for item in rec_text.split("\n") if item.strip()]
        elif ";" in rec_text:
            data["recommendations"] = [item.strip() for item in rec_text.split(";") if item.strip()]
        else:
            data["recommendations"] = [rec_text.strip()]
    
    # Ensure confidence is a float
    if "confidence" in data:
        data["confidence"] = float(data["confidence"])
    
    return data


def _normalize_rca_response(data: dict) -> dict:
    """
    Normalize LLM RCA response to match RootCauseAnalysis schema.
    """
    if "hypotheses" not in data:
        raise ValueError("RCA response must contain 'hypotheses' key")
    
    hypotheses = data["hypotheses"]
    if not isinstance(hypotheses, list):
        hypotheses = [hypotheses]
    
    normalized_hypotheses = []
    for hyp in hypotheses:
        if isinstance(hyp, dict):
            # Convert category to lowercase
            if "category" in hyp:
                hyp["category"] = hyp["category"].lower()
            
            # Convert evidence to list if it's a string
            if "evidence" in hyp and isinstance(hyp["evidence"], str):
                evidence_text = hyp["evidence"]
                if "\n" in evidence_text:
                    hyp["evidence"] = [item.strip() for item in evidence_text.split("\n") if item.strip()]
                elif ";" in evidence_text:
                    hyp["evidence"] = [item.strip() for item in evidence_text.split(";") if item.strip()]
                else:
                    hyp["evidence"] = [evidence_text.strip()]
            
            # Convert recommended_checks to list if it's a string
            if "recommended_checks" in hyp and isinstance(hyp["recommended_checks"], str):
                checks_text = hyp["recommended_checks"]
                if "\n" in checks_text:
                    hyp["recommended_checks"] = [item.strip() for item in checks_text.split("\n") if item.strip()]
                elif ";" in checks_text:
                    hyp["recommended_checks"] = [item.strip() for item in checks_text.split(";") if item.strip()]
                else:
                    hyp["recommended_checks"] = [checks_text.strip()]
            
            # Ensure confidence is a float
            if "confidence" in hyp:
                hyp["confidence"] = float(hyp["confidence"])
            
            normalized_hypotheses.append(hyp)
    
    data["hypotheses"] = normalized_hypotheses
    return data


class AIInvestigator:
    """
    Use an LLM to investigate evidence produced
    by the deterministic ML regression system.
    """

    def __init__(
        self,
        model: str,
    ) -> None:

        api_key = os.getenv(
            "OPENROUTER_API_KEY"
        )

        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not configured."
            )

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        self.model = model

    def analyze_root_cause(
        self,
        evidence: dict,
    ) -> RootCauseAnalysis:
        """
        Analyze the core reason behind a performance drop.
        """
        prompt = build_rca_prompt(evidence)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": RCA_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.1,
        )

        content = response.choices[0].message.content

        if not content:
            raise ValueError(
                "LLM returned an empty response."
            )

        data = _extract_json(content)
        
        # Normalize the response to match schema
        data = _normalize_rca_response(data)

        return RootCauseAnalysis.model_validate(data)    

    def investigate(
        self,
        evidence: dict,
    ) -> AIInvestigation:
        """
        Investigate a detected model regression.
        """

        prompt = build_investigation_prompt(
            evidence
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.1,
        )

        content = response.choices[0].message.content

        if not content:
            raise ValueError(
                "LLM returned an empty response."
            )

        data = _extract_json(content)
        
        # Normalize the response to match schema
        data = _normalize_investigation_response(data)

        return AIInvestigation.model_validate(
            data
        )
