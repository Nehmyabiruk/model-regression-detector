export interface MetricComparison {
  metric_name: string;
  baseline_value: number;
  candidate_value: number;
  difference: number;
  relative_change?: number;
  threshold: number;
  regression: boolean;
}

export interface DriftResult {
  feature: string;
  score: number;
  threshold: number;
  drift_detected: boolean;
}

export interface SegmentResult {
  segment: string;
  metric: string;
  baseline: number;
  candidate: number;
  difference: number;
  regression: boolean;
}

export interface Investigation {
  severity: string;
  root_cause: string;
  evidence: string[];
  affected_areas: string[];
  recommendations: string[];
  confidence: number;
}

export interface RootCauseHypothesis {
  cause: string;
  category: string;
  confidence: number;
  evidence: string[];
  reasoning: string;
  recommended_checks: string[];
}

export interface RootCause {
  hypotheses: RootCauseHypothesis[];
}

export interface Recommendation {
  action: string;
  reason: string;
  priority: string;
  expected_outcome: string;
  confidence: number;
}

export interface Recommendations {
  recommendations: Recommendation[];
}

export interface InvestigationResponse {
  investigation: Investigation;
  root_cause: RootCause;
  recommendations: Recommendations;
  agent_analysis: string;
}

export interface RegressionReport {
  report_id: string;
  model_name: string;
  baseline_version: string;
  candidate_version: string;
  dataset_name: string;
  task_type?: string;
  status: string;
  performance: Record<string, MetricComparison>;
  drift: DriftResult[];
  segments: SegmentResult[];
  summary: string;
}