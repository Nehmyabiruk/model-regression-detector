import type {
  InvestigationResponse,
  RegressionReport,
} from "../types/mrd";

const baseURL =
  import.meta.env.VITE_API_URL ||
  "http://127.0.0.1:8000";

const api = {
  async post<T>(path: string, data: unknown): Promise<{ data: T }> {
    const response = await fetch(`${baseURL}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `API error: ${response.statusText}`);
    }
    return { data: await response.json() };
  },
  async postFormData<T>(path: string, formData: FormData): Promise<{ data: T }> {
    const response = await fetch(`${baseURL}${path}`, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `API error: ${response.statusText}`);
    }
    return { data: await response.json() };
  },
  async get(path: string) {
    const response = await fetch(`${baseURL}${path}`);
    if (!response.ok) throw new Error(`API error: ${response.statusText}`);
    return await response.json();
  },
};

export async function investigateRegression(
  report: RegressionReport
): Promise<InvestigationResponse> {
  const response = await api.post<InvestigationResponse>(
    "/ai/investigate",
    {
      report,
    }
  );

  return response.data;
}

export async function runEvaluation(
  modelName: string,
  baselineVersion: string,
  candidateVersion: string,
  datasetVersion: string,
  targetColumn: string,
  modelType: string,
  baselineModel: File,
  candidateModel: File,
  evaluationDataset: File,
  timeColumn?: string
): Promise<{ report: RegressionReport }> {
  const formData = new FormData();
  formData.append("model_name", modelName);
  formData.append("baseline_version", baselineVersion);
  formData.append("candidate_version", candidateVersion);
  formData.append("dataset_version", datasetVersion);
  formData.append("target_column", targetColumn);
  formData.append("model_type", modelType);
  if (timeColumn) {
    formData.append("time_column", timeColumn);
  }
  formData.append("baseline_model", baselineModel);
  formData.append("candidate_model", candidateModel);
  formData.append("evaluation_dataset", evaluationDataset);

  const response = await api.postFormData<{ report: RegressionReport }>(
    "/evaluations/run",
    formData
  );

  return response.data;
}

export async function checkHealth() {
  const response = await api.get("/health");
  return response;
}

export default api;