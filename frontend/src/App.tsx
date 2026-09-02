import { useState } from "react";
import {
  Activity,
  AlertTriangle,
  Brain,
  ChevronRight,
  Database,
  Gauge,
  Menu,
  Plus,
  Search,
  ShieldAlert,
  Sparkles,
  TrendingDown,
  Users,
  X,
  UploadCloud,
  CheckCircle2,
  AlertCircle,
} from "lucide-react";

import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type {
  InvestigationResponse,
  RegressionReport,
} from "./types/mrd";

import { investigateRegression, runEvaluation } from "./services/api";

type PageType = "home" | "new-evaluation" | "overview" | "investigations" | "models" | "drift" | "segments";

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [currentPage, setCurrentPage] = useState<PageType>("home");
  const [report, setReport] = useState<RegressionReport | null>(null);
  const [investigation, setInvestigation] = useState<InvestigationResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const metricData = report
    ? Object.values(report.performance).map((metric: any) => ({
        name: metric.metric_name.replace("_", " ").toUpperCase(),
        baseline: metric.baseline_value,
        candidate: metric.candidate_value,
      }))
    : [];

  return (
    <div className="min-h-screen bg-[#070b14] text-gray-100">
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/60 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside className={`fixed left-0 top-0 z-50 h-screen w-72 border-r border-white/10 bg-[#0a0f1c] transition-transform duration-300 ${sidebarOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"}`}>
        <div className="flex h-20 items-center justify-between border-b border-white/10 px-6">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-500/15 ring-1 ring-indigo-400/30">
              <Brain className="h-5 w-5 text-indigo-400" />
            </div>
            <div>
              <h1 className="font-bold tracking-tight">MRD</h1>
              <p className="text-[10px] uppercase tracking-[0.2em] text-gray-500">Model Regression</p>
            </div>
          </div>
          <button className="lg:hidden" onClick={() => setSidebarOpen(false)}>
            <X className="h-5 w-5 text-gray-400" />
          </button>
        </div>

        <nav className="space-y-2 p-4">
          <button
            onClick={() => {
              setCurrentPage("home");
              setSidebarOpen(false);
            }}
            className={`flex w-full items-center gap-3 rounded-xl px-4 py-3 text-sm transition ${
              currentPage === "home"
                ? "bg-indigo-500/10 text-indigo-300 ring-1 ring-indigo-400/10"
                : "text-gray-500 hover:bg-white/3 hover:text-gray-300"
            }`}
          >
            <Activity />
            Home
          </button>

          <button
            onClick={() => {
              setCurrentPage("new-evaluation");
              setSidebarOpen(false);
            }}
            className={`flex w-full items-center gap-3 rounded-xl px-4 py-3 text-sm transition ${
              currentPage === "new-evaluation"
                ? "bg-indigo-500/10 text-indigo-300 ring-1 ring-indigo-400/10"
                : "text-gray-500 hover:bg-white/3 hover:text-gray-300"
            }`}
          >
            <Plus />
            New Evaluation
          </button>

          {report && (
            <>
              <div className="my-4 h-px bg-white/10" />

              {[
                { label: "Overview", icon: <Activity />, page: "overview" as PageType },
                { label: "Investigations", icon: <ShieldAlert />, page: "investigations" as PageType },
                { label: "Models", icon: <Database />, page: "models" as PageType },
                { label: "Drift", icon: <TrendingDown />, page: "drift" as PageType },
                { label: "Segments", icon: <Users />, page: "segments" as PageType },
              ].map(({ label, icon, page }) => (
                <button
                  key={page}
                  onClick={() => {
                    setCurrentPage(page);
                    setSidebarOpen(false);
                  }}
                  className={`flex w-full items-center gap-3 rounded-xl px-4 py-3 text-sm transition ${
                    currentPage === page
                      ? "bg-indigo-500/10 text-indigo-300 ring-1 ring-indigo-400/10"
                      : "text-gray-500 hover:bg-white/3 hover:text-gray-300"
                  }`}
                >
                  {icon}
                  {label}
                </button>
              ))}
            </>
          )}
        </nav>

        <div className="absolute bottom-5 left-4 right-4 rounded-2xl border border-indigo-400/20 bg-indigo-500/5 p-4">
          <div className="mb-2 flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-indigo-400" />
            <span className="text-xs font-semibold">AI Engine</span>
          </div>
          <p className="text-xs leading-5 text-gray-500">Real-time regression detection and AI-powered root-cause analysis.</p>
          <div className="mt-3 flex items-center gap-2 text-xs text-emerald-400">
            <span className="h-2 w-2 rounded-full bg-emerald-400" />
            Operational
          </div>
        </div>
      </aside>

      {/* Main */}
      <main className="lg:ml-72">
        <header className="sticky top-0 z-30 flex h-20 items-center justify-between border-b border-white/10 bg-[#070b14]/90 px-5 backdrop-blur-xl lg:px-8">
          <div className="flex items-center gap-4">
            <button className="lg:hidden" onClick={() => setSidebarOpen(true)}>
              <Menu className="h-6 w-6" />
            </button>
            <div>
              <p className="text-xs uppercase tracking-widest text-gray-500">Model Operations</p>
              <h2 className="text-lg font-semibold">
                {currentPage === "home"
                  ? "Welcome"
                  : currentPage === "new-evaluation"
                  ? "New Evaluation"
                  : currentPage.charAt(0).toUpperCase() + currentPage.slice(1)}
              </h2>
            </div>
          </div>
          <div className="hidden items-center gap-3 md:flex">
            <div className="flex items-center gap-2 rounded-xl border border-white/10 bg-white/3 px-4 py-2">
              <Search className="h-4 w-4 text-gray-500" />
              <span className="text-sm text-gray-500">Search</span>
              <kbd className="ml-5 rounded bg-white/5 px-2 py-0.5 text-[10px] text-gray-500">/</kbd>
            </div>
            <div className="flex items-center gap-2 rounded-xl border border-emerald-400/20 bg-emerald-400/5 px-3 py-2 text-xs text-emerald-400">
              <span className="h-2 w-2 rounded-full bg-emerald-400" />
              System healthy
            </div>
          </div>
        </header>

        <div className="p-5 lg:p-8">
          {currentPage === "home" && <HomePage onStartEvaluation={() => setCurrentPage("new-evaluation")} />}

          {currentPage === "new-evaluation" && (
            <NewEvaluationPage
              loading={loading}
              error={error}
              onEvaluationComplete={(newReport) => {
                setReport(newReport);
                setInvestigation(null);
                setError("");
                setCurrentPage("overview");
              }}
              onError={(err) => setError(err)}
            />
          )}

          {report && currentPage === "overview" && (
            <OverviewPage
              report={report}
              investigation={investigation}
              loading={loading}
              error={error}
              metricData={metricData}
              onRunInvestigation={async () => {
                setLoading(true);
                setError("");
                try {
                  const result = await investigateRegression(report);
                  setInvestigation(result);
                } catch (err) {
                  const error = err as { response?: { data?: { detail?: string } }; message?: string };
                  setError(error?.response?.data?.detail || error?.message || "Investigation failed.");
                } finally {
                  setLoading(false);
                }
              }}
            />
          )}

          {report && currentPage === "investigations" && <InvestigationsPage investigation={investigation} />}
          {report && currentPage === "models" && <ModelsPage report={report} />}
          {report && currentPage === "drift" && <DriftPage report={report} />}
          {report && currentPage === "segments" && <SegmentsPage report={report} />}

          {!report && currentPage !== "home" && currentPage !== "new-evaluation" && (
            <div className="text-center text-gray-500 py-12">
              <AlertCircle className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p>No evaluation loaded. Please run a new evaluation first.</p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

function HomePage({ onStartEvaluation }: { onStartEvaluation: () => void }) {
  return (
    <div className="max-w-4xl">
      <section className="mb-12">
        <div className="mb-6 flex items-center gap-3">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-indigo-500/10">
            <Brain className="h-6 w-6 text-indigo-400" />
          </div>
          <div>
            <h1 className="text-4xl font-bold tracking-tight">Model Regression Detector</h1>
            <p className="text-sm text-gray-500 mt-1">AI-powered model evaluation and root-cause analysis</p>
          </div>
        </div>
        <p className="text-gray-400 text-lg leading-7 max-w-2xl">
          Upload your baseline and candidate models, provide your evaluation dataset, and let MRD automatically detect regressions, analyze feature drift, and use AI to identify root causes.
        </p>
      </section>

      <section className="grid gap-4 md:grid-cols-2 lg:grid-cols-3 mb-12">
        <FeatureCard
          icon={<UploadCloud className="h-6 w-6" />}
          title="Upload Models"
          description="Support for joblib/pickle serialized models"
        />
        <FeatureCard
          icon={<Database className="h-6 w-6" />}
          title="Real Evaluation"
          description="Actual metrics (ROC AUC, precision, recall, etc.)"
        />
        <FeatureCard
          icon={<TrendingDown className="h-6 w-6" />}
          title="Drift Detection"
          description="Feature distribution analysis using PSI"
        />
        <FeatureCard
          icon={<AlertTriangle className="h-6 w-6" />}
          title="Regression Detection"
          description="Configurable threshold-based detection"
        />
        <FeatureCard
          icon={<Sparkles className="h-6 w-6" />}
          title="AI Investigation"
          description="LLM-powered root cause analysis"
        />
        <FeatureCard
          icon={<CheckCircle2 className="h-6 w-6" />}
          title="Recommendations"
          description="Actionable next steps and insights"
        />
      </section>

      <section className="rounded-2xl border border-indigo-400/20 bg-indigo-500/5 p-8 text-center">
        <h2 className="text-2xl font-bold mb-3">Ready to evaluate your models?</h2>
        <p className="text-gray-400 mb-6">Start by creating a new evaluation. Upload your models and dataset to begin.</p>
        <button
          onClick={onStartEvaluation}
          className="inline-flex items-center gap-2 rounded-xl bg-indigo-500 px-8 py-4 text-lg font-semibold shadow-lg shadow-indigo-500/20 transition hover:bg-indigo-400"
        >
          <Plus className="h-5 w-5" />
          Create New Evaluation
        </button>
      </section>
    </div>
  );
}

function FeatureCard({ icon, title, description }: { icon: React.ReactNode; title: string; description: string }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/1 p-6 hover:border-indigo-400/30 transition">
      <div className="mb-3 text-indigo-400">{icon}</div>
      <h3 className="font-semibold mb-2">{title}</h3>
      <p className="text-sm text-gray-500">{description}</p>
    </div>
  );
}

function NewEvaluationPage({
  loading,
  error,
  onEvaluationComplete,
  onError,
}: {
  loading: boolean;
  error: string;
  onEvaluationComplete: (report: RegressionReport) => void;
  onError: (error: string) => void;
}) {
  const [formData, setFormData] = useState({
    modelName: "",
    baselineVersion: "",
    candidateVersion: "",
    datasetVersion: "",
    targetColumn: "",
    modelType: "classification",
    timeColumn: "",
  });

  const [files, setFiles] = useState<{
    baselineModel: File | null;
    candidateModel: File | null;
    dataset: File | null;
  }>({
    baselineModel: null,
    candidateModel: null,
    dataset: null,
  });

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();

    if (!files.baselineModel || !files.candidateModel || !files.dataset) {
      onError("Please upload all required files");
      return;
    }

    if (!formData.modelName || !formData.targetColumn) {
      onError("Please fill in all required fields");
      return;
    }

    if (formData.modelType === "timeseries" && !formData.timeColumn) {
      onError("Please specify the time/date column for time-series evaluation");
      return;
    }

    try {
      const result = await runEvaluation(
        formData.modelName,
        formData.baselineVersion,
        formData.candidateVersion,
        formData.datasetVersion,
        formData.targetColumn,
        formData.modelType,
        files.baselineModel,
        files.candidateModel,
        files.dataset,
        formData.modelType === "timeseries" ? formData.timeColumn : undefined
      );

      onEvaluationComplete(result.report);
    } catch (err) {
      const error = err as { response?: { data?: { detail?: string } }; message?: string };
      onError(error?.response?.data?.detail || error?.message || "Evaluation failed. Please check your files and try again.");
    }
  }

  return (
    <div className="max-w-2xl">
      <form onSubmit={handleSubmit} className="space-y-6">
        {error && (
          <div className="rounded-2xl border border-red-400/20 bg-red-400/5 p-4 text-sm text-red-300 flex gap-3">
            <AlertTriangle className="h-5 w-5 shrink-0 mt-0.5" />
            <div>{error}</div>
          </div>
        )}

        <Panel title="Model Configuration" subtitle="Basic information about your models">
          <div className="grid gap-4">
            <FormField label="Model Name" required>
              <input
                type="text"
                placeholder="e.g., credit-risk"
                value={formData.modelName}
                onChange={(e) => setFormData({ ...formData, modelName: e.target.value })}
                className="w-full rounded-xl border border-white/10 bg-white/3 px-4 py-3 text-white placeholder-gray-500 focus:border-indigo-400/50 outline-none transition"
              />
            </FormField>

            <div className="grid gap-4 md:grid-cols-3">
              <FormField label="Baseline Version" required>
                <input
                  type="text"
                  placeholder="e.g., 2.2.0"
                  value={formData.baselineVersion}
                  onChange={(e) => setFormData({ ...formData, baselineVersion: e.target.value })}
                  className="w-full rounded-xl border border-white/10 bg-white/3 px-4 py-3 text-white placeholder-gray-500 focus:border-indigo-400/50 outline-none transition"
                />
              </FormField>

              <FormField label="Candidate Version" required>
                <input
                  type="text"
                  placeholder="e.g., 2.3.0"
                  value={formData.candidateVersion}
                  onChange={(e) => setFormData({ ...formData, candidateVersion: e.target.value })}
                  className="w-full rounded-xl border border-white/10 bg-white/3 px-4 py-3 text-white placeholder-gray-500 focus:border-indigo-400/50 outline-none transition"
                />
              </FormField>

              <FormField label="Dataset Version" required>
                <input
                  type="text"
                  placeholder="e.g., eval-v1"
                  value={formData.datasetVersion}
                  onChange={(e) => setFormData({ ...formData, datasetVersion: e.target.value })}
                  className="w-full rounded-xl border border-white/10 bg-white/3 px-4 py-3 text-white placeholder-gray-500 focus:border-indigo-400/50 outline-none transition"
                />
              </FormField>
            </div>
          </div>
        </Panel>

        <Panel title="Dataset Configuration" subtitle="Specify how to evaluate your data">
          <div className="grid gap-4">
            <FormField label="Target Column" required>
              <input
                type="text"
                placeholder="e.g., y or target"
                value={formData.targetColumn}
                onChange={(e) => setFormData({ ...formData, targetColumn: e.target.value })}
                className="w-full rounded-xl border border-white/10 bg-white/3 px-4 py-3 text-white placeholder-gray-500 focus:border-indigo-400/50 outline-none transition"
              />
            </FormField>

            <FormField label="Model Type" required>
              <select
                value={formData.modelType}
                onChange={(e) => setFormData({ ...formData, modelType: e.target.value })}
                className="w-full rounded-xl border border-white/10 bg-white/3 px-4 py-3 text-white focus:border-indigo-400/50 outline-none transition"
              >
                <option value="classification">Classification (Binary & Multiclass)</option>
                <option value="regression">Regression</option>
                <option value="timeseries">Time Series Forecasting</option>
              </select>
            </FormField>

            {formData.modelType === "timeseries" && (
              <FormField label="Time Column" required>
                <input
                  type="text"
                  placeholder="e.g., date, timestamp"
                  value={formData.timeColumn}
                  onChange={(e) => setFormData({ ...formData, timeColumn: e.target.value })}
                  className="w-full rounded-xl border border-white/10 bg-white/3 px-4 py-3 text-white placeholder-gray-500 focus:border-indigo-400/50 outline-none transition"
                />
              </FormField>
            )}
          </div>
        </Panel>

        <Panel title="Upload Files" subtitle="Your models (joblib/pickle) and evaluation dataset (CSV)">
          <div className="space-y-4">
            <FileUploadField
              label="Baseline Model"
              hint="joblib or pickle format"
              accept=".joblib,.pkl,.pickle"
              onChange={(file) => setFiles({ ...files, baselineModel: file })}
              fileName={files.baselineModel?.name}
            />

            <FileUploadField
              label="Candidate Model"
              hint="joblib or pickle format"
              accept=".joblib,.pkl,.pickle"
              onChange={(file) => setFiles({ ...files, candidateModel: file })}
              fileName={files.candidateModel?.name}
            />

            <FileUploadField
              label="Evaluation Dataset"
              hint="CSV format with target column"
              accept=".csv"
              onChange={(file) => setFiles({ ...files, dataset: file })}
              fileName={files.dataset?.name}
            />
          </div>
        </Panel>

        <button
          type="submit"
          disabled={loading}
          className="w-full rounded-xl bg-indigo-500 px-6 py-4 font-semibold shadow-lg shadow-indigo-500/20 transition hover:bg-indigo-400 disabled:cursor-not-allowed disabled:opacity-60 flex items-center justify-center gap-2"
        >
          {loading && <div className="h-5 w-5 animate-spin rounded-full border-2 border-white/30 border-t-white" />}
          {loading ? "Running Evaluation..." : "Run Evaluation"}
        </button>
      </form>
    </div>
  );
}

function FileUploadField({
  label,
  hint,
  accept,
  onChange,
  fileName,
}: {
  label: string;
  hint: string;
  accept: string;
  onChange: (file: File | null) => void;
  fileName?: string;
}) {
  return (
    <div>
      <label className="block text-sm font-medium text-gray-300 mb-2">{label}</label>
      <div className="relative">
        <input
          type="file"
          accept={accept}
          onChange={(e) => onChange(e.target.files?.[0] || null)}
          className="absolute inset-0 opacity-0 cursor-pointer"
        />
        <div className="rounded-xl border-2 border-dashed border-white/10 bg-white/2 p-6 text-center hover:border-indigo-400/50 transition">
          <UploadCloud className="h-8 w-8 text-gray-400 mx-auto mb-2" />
          {fileName ? (
            <>
              <p className="font-medium text-white">{fileName}</p>
              <p className="text-xs text-gray-500">Click to change</p>
            </>
          ) : (
            <>
              <p className="font-medium text-gray-300">Click to upload</p>
              <p className="text-xs text-gray-500">{hint}</p>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function FormField({ label, required, children }: { label: string; required?: boolean; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-sm font-medium text-gray-300 mb-2">
        {label}
        {required && <span className="text-red-400"> *</span>}
      </label>
      {children}
    </div>
  );
}

function Panel({
  title,
  subtitle,
  className = "",
  children,
}: {
  title: string;
  subtitle?: string;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <div className={`rounded-2xl border border-white/10 bg-white/1 p-6 ${className}`}>
      <div className="mb-6">
        <h3 className="text-lg font-semibold">{title}</h3>
        {subtitle && <p className="mt-1 text-sm text-gray-500">{subtitle}</p>}
      </div>
      {children}
    </div>
  );
}

function OverviewPage({
  report,
  investigation,
  loading,
  error,
  metricData,
  onRunInvestigation,
}: any) {
  return (
    <>
      <section className="mb-8 flex flex-col justify-between gap-5 md:flex-row md:items-end">
        <div>
          <div className="mb-3 flex items-center gap-2 text-xs text-gray-500">
            <span>Models</span>
            <ChevronRight className="h-3 w-3" />
            <span className="text-gray-300">{report.model_name}</span>
          </div>
          <h1 className="text-3xl font-bold tracking-tight">Model health</h1>
          <p className="mt-2 max-w-2xl text-sm text-gray-500">
            Monitor model performance, detect regressions, and investigate failures using AI-powered root-cause analysis.
          </p>
        </div>
        <button
          onClick={onRunInvestigation}
          disabled={loading}
          className="flex items-center justify-center gap-2 rounded-xl bg-indigo-500 px-5 py-3 text-sm font-semibold shadow-lg shadow-indigo-500/20 transition hover:bg-indigo-400 disabled:cursor-not-allowed disabled:opacity-60"
        >
          <Sparkles className="h-4 w-4" />
          {loading ? "Investigating..." : "Run AI Investigation"}
        </button>
      </section>

      {error && (
        <div className="mb-6 rounded-2xl border border-red-400/20 bg-red-400/5 p-4 text-sm text-red-300">{error}</div>
      )}

      <section className="mb-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          icon={<ShieldAlert />}
          title="Current status"
          value={report.status.toUpperCase()}
          description={
            report.status === "regression"
              ? "Candidate requires investigation"
              : "Models perform similarly"
          }
          danger={report.status === "regression"}
        />
        <StatCard icon={<Gauge />} title="Model" value={report.model_name} description={`v${report.candidate_version} candidate`} />
        <StatCard
          icon={<TrendingDown />}
          title="Metric regressions"
          value={Object.values(report.performance).filter((m: any) => m.regression).length}
          description="Across monitored metrics"
          danger={Object.values(report.performance).some((m: any) => m.regression)}
        />
        <StatCard
          icon={<AlertTriangle />}
          title="Drift signals"
          value={report.drift.filter((d: any) => d.severity === "high").length}
          description="Features with high drift"
          warning={report.drift.some((d: any) => d.severity === "high")}
        />
      </section>

      <section className="grid gap-6 xl:grid-cols-3">
        <Panel title="Performance comparison" subtitle="Baseline vs candidate" className="xl:col-span-2">
          {metricData.length > 0 ? (
            <>
              <div className="h-80 w-full">
                <ResponsiveContainer>
                  <BarChart data={metricData} barGap={10}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                    <XAxis dataKey="name" tick={{ fill: "#6b7280", fontSize: 11 }} axisLine={false} tickLine={false} />
                    <YAxis domain={[0, "auto"]} tick={{ fill: "#6b7280", fontSize: 11 }} axisLine={false} tickLine={false} />
                    <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid rgba(255,255,255,.1)", borderRadius: "12px" }} />
                    <Bar dataKey="baseline" name="Baseline" fill="#6366f1" radius={[5, 5, 0, 0]} />
                    <Bar dataKey="candidate" name="Candidate" fill="#f43f5e" radius={[5, 5, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="grid gap-3 md:grid-cols-2">
                {Object.values(report.performance).map((metric: any) => (
                  <MetricRow key={metric.metric_name} metric={metric} />
                ))}
              </div>
            </>
          ) : (
            <div className="h-80 flex items-center justify-center text-gray-500">No metrics available</div>
          )}
        </Panel>

        <Panel title="Model context" subtitle="Deployment metadata">
          <div className="space-y-5">
            <InfoRow label="Model" value={report.model_name} />
            <InfoRow label="Task Type" value={report.task_type || "classification"} />
            <InfoRow label="Baseline" value={`v${report.baseline_version}`} />
            <InfoRow label="Candidate" value={`v${report.candidate_version}`} />
            <InfoRow label="Dataset" value={report.dataset_name} />
            <InfoRow label="Report" value={report.report_id} />
            {report.status === "regression" && (
              <div className="rounded-xl border border-red-400/20 bg-red-400/5 p-4">
                <div className="flex items-center gap-2 text-sm font-semibold text-red-300">
                  <AlertTriangle className="h-4 w-4" />
                  Deployment blocked
                </div>
                <p className="mt-2 text-xs leading-5 text-gray-500">Candidate model should be investigated before promotion.</p>
              </div>
            )}
          </div>
        </Panel>
      </section>

      {investigation && (
        <section className="mt-6 space-y-6">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-500/10">
              <Sparkles className="h-5 w-5 text-indigo-400" />
            </div>
            <div>
              <h2 className="font-semibold">AI Investigation</h2>
              <p className="text-xs text-gray-500">Generated from regression evidence</p>
            </div>
          </div>

          <Panel title="AI assessment" subtitle="Automated investigation">
            <div className="flex flex-col gap-6 md:flex-row">
              <div className="flex-1">
                <div className="mb-3 flex items-center gap-3">
                  <span className="rounded-full bg-red-400/10 px-3 py-1 text-xs font-semibold uppercase text-red-300">
                    {investigation.investigation.severity}
                  </span>
                  <span className="text-xs text-gray-500">Confidence {Math.round(investigation.investigation.confidence * 100)}%</span>
                </div>
                <p className="text-sm leading-7 text-gray-300">{investigation.investigation.root_cause}</p>
              </div>
              <div className="w-full rounded-2xl border border-indigo-400/15 bg-indigo-400/5 p-5 md:max-w-xs">
                <div className="mb-2 flex items-center gap-2 text-indigo-300">
                  <Brain className="h-4 w-4" />
                  <span className="text-sm font-semibold">AI confidence</span>
                </div>
                <div className="text-3xl font-bold">{Math.round(investigation.investigation.confidence * 100)}%</div>
                <div className="mt-3 h-2 overflow-hidden rounded-full bg-white/5">
                  <div className="h-full rounded-full bg-indigo-400" style={{ width: `${investigation.investigation.confidence * 100}%` }} />
                </div>
              </div>
            </div>
          </Panel>

          <Panel title="Root-cause hypotheses" subtitle="Ranked explanations generated by AI">
            <div className="grid gap-4 lg:grid-cols-2">
              {investigation.root_cause.hypotheses.map((hypothesis: any, index: number) => (
                <div key={index} className="rounded-2xl border border-white/10 bg-white/2 p-5">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <span className="text-[10px] uppercase tracking-widest text-gray-600">Hypothesis {index + 1}</span>
                      <h3 className="mt-2 text-sm font-semibold leading-6">{hypothesis.cause}</h3>
                    </div>
                    <span className="shrink-0 rounded-full bg-indigo-400/10 px-3 py-1 text-xs text-indigo-300">{Math.round(hypothesis.confidence * 100)}%</span>
                  </div>
                  <p className="mt-4 text-xs leading-6 text-gray-500">{hypothesis.reasoning}</p>
                </div>
              ))}
            </div>
          </Panel>

          <Panel title="AI recommendations" subtitle="Suggested remediation actions">
            <div className="space-y-3">
              {investigation.recommendations.recommendations.map((rec: any, index: number) => (
                <div key={index} className="rounded-2xl border border-white/10 bg-white/2 p-5">
                  <div className="flex gap-4">
                    <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-indigo-500/10 text-sm font-bold text-indigo-400">{index + 1}</div>
                    <div className="flex-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <h3 className="text-sm font-semibold">{rec.action}</h3>
                        <span className="rounded-full bg-white/5 px-2 py-1 text-[10px] uppercase text-gray-500">{rec.priority}</span>
                      </div>
                      <p className="mt-2 text-xs leading-6 text-gray-500">{rec.reason}</p>
                      <p className="mt-3 text-xs text-gray-400">
                        <span className="font-semibold text-gray-300">Expected:</span> {rec.expected_outcome}
                      </p>
                    </div>
                    <div className="hidden text-right sm:block">
                      <p className="text-xs text-gray-600">Confidence</p>
                      <p className="mt-1 font-semibold text-indigo-300">{Math.round(rec.confidence * 100)}%</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Panel>
        </section>
      )}
    </>
  );
}

function InvestigationsPage({ investigation }: any) {
  return (
    <div>
      <h1 className="text-3xl font-bold tracking-tight mb-8">Investigations</h1>
      {investigation ? (
        <Panel title="Investigation Results" subtitle="Latest investigation">
          <div className="text-sm text-gray-300 space-y-3">
            <div>
              <strong>Status:</strong> {investigation.investigation.severity}
            </div>
            <div>
              <strong>Confidence:</strong> {Math.round(investigation.investigation.confidence * 100)}%
            </div>
            <div>
              <strong>Summary:</strong> {investigation.investigation.root_cause}
            </div>
          </div>
        </Panel>
      ) : (
        <div className="text-center text-gray-500">No investigations yet. Run an investigation from Overview.</div>
      )}
    </div>
  );
}

function ModelsPage({ report }: any) {
  return (
    <div>
      <h1 className="text-3xl font-bold tracking-tight mb-8">Models</h1>
      <Panel title="Model Information" subtitle={report.model_name}>
        <div className="space-y-4">
          <InfoRow label="Model Name" value={report.model_name} />
          <InfoRow label="Baseline Version" value={`v${report.baseline_version}`} />
          <InfoRow label="Candidate Version" value={`v${report.candidate_version}`} />
          <InfoRow label="Dataset" value={report.dataset_name} />
          <InfoRow label="Status" value={report.status.toUpperCase()} />
        </div>
      </Panel>
    </div>
  );
}

function DriftPage({ report }: any) {
  return (
    <div>
      <h1 className="text-3xl font-bold tracking-tight mb-8">Drift Detection</h1>
      <Panel title="Feature Drift" subtitle="Distribution changes detected between evaluation datasets">
        {report.drift.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full min-w-150 text-left">
              <thead>
                <tr className="border-b border-white/10 text-xs uppercase tracking-wider text-gray-600">
                  <th className="pb-4">Feature</th>
                  <th className="pb-4">Score (PSI)</th>
                  <th className="pb-4">Method</th>
                  <th className="pb-4">Severity</th>
                </tr>
              </thead>
              <tbody>
                {report.drift.map((item: any) => (
                  <tr key={item.feature_name} className="border-b border-white/5">
                    <td className="py-4 font-medium">{item.feature_name}</td>
                    <td className="py-4">
                      <div className="flex items-center gap-3">
                        <div className="h-2 w-32 overflow-hidden rounded-full bg-white/5">
                          <div
                            className={`h-full rounded-full ${
                              item.severity === "high" ? "bg-red-400" : item.severity === "moderate" ? "bg-yellow-400" : "bg-emerald-400"
                            }`}
                            style={{ width: `${Math.min(item.score * 100, 100)}%` }}
                          />
                        </div>
                        <span className="text-sm font-mono">{item.score.toFixed(3)}</span>
                      </div>
                    </td>
                    <td className="py-4 text-sm text-gray-500">{item.method}</td>
                    <td className="py-4">
                      <span
                        className={`rounded-full px-3 py-1 text-xs font-medium ${
                          item.severity === "high"
                            ? "bg-red-400/10 text-red-300"
                            : item.severity === "moderate"
                            ? "bg-yellow-400/10 text-yellow-300"
                            : "bg-emerald-400/10 text-emerald-300"
                        }`}
                      >
                        {item.severity}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center text-gray-500 py-8">No drift data available</div>
        )}
      </Panel>
    </div>
  );
}

function SegmentsPage({ report }: any) {
  return (
    <div>
      <h1 className="text-3xl font-bold tracking-tight mb-8">Segments</h1>
      {report.segments.length > 0 ? (
        <Panel title="Segment Regression" subtitle="Performance degradation by population">
          <div className="grid gap-4 md:grid-cols-2">
            {report.segments.map((segment: any) => (
              <div key={segment.segment_name} className="rounded-2xl border border-red-400/15 bg-red-400/3 p-5">
                <div className="flex items-start justify-between">
                  <div>
                    <p className="font-semibold">{segment.segment_name}</p>
                    <p className="mt-1 text-xs text-gray-500">{segment.metric_name}</p>
                  </div>
                  <span className={`rounded-full px-3 py-1 text-xs ${segment.regression ? "bg-red-400/10 text-red-300" : "bg-emerald-400/10 text-emerald-300"}`}>
                    {segment.regression ? "Regression" : "Normal"}
                  </span>
                </div>
                <div className="mt-5 grid grid-cols-3 gap-3">
                  <MiniMetric label="Baseline" value={segment.baseline_value} />
                  <MiniMetric label="Candidate" value={segment.candidate_value} />
                  <MiniMetric label="Change" value={segment.difference} negative />
                </div>
              </div>
            ))}
          </div>
        </Panel>
      ) : (
        <Panel title="Segment Regression" subtitle="No segments available">
          <div className="text-center text-gray-500">No segment data available</div>
        </Panel>
      )}
    </div>
  );
}

function StatCard({
  icon,
  title,
  value,
  description,
  danger = false,
  warning = false,
}: any) {
  return (
    <div className={`rounded-2xl border p-6 ${danger ? "border-red-400/20 bg-red-400/5" : warning ? "border-yellow-400/20 bg-yellow-400/5" : "border-white/10 bg-white/1"}`}>
      <div className="flex items-start justify-between">
        <div
          className={`flex h-10 w-10 items-center justify-center rounded-xl ${danger ? "bg-red-400/10 text-red-300" : warning ? "bg-yellow-400/10 text-yellow-300" : "bg-indigo-400/10 text-indigo-400"}`}
        >
          {icon}
        </div>
      </div>
      <p className={`mt-4 text-3xl font-bold ${danger ? "text-red-300" : warning ? "text-yellow-300" : "text-white"}`}>{value}</p>
      <p className="mt-1 text-xs text-gray-500">{title}</p>
      <p className="mt-2 text-xs text-gray-400">{description}</p>
    </div>
  );
}

function MetricRow({ metric }: any) {
  return (
    <div className="rounded-xl border border-white/5 p-4">
      <div className="flex items-center justify-between">
        <p className="text-sm font-medium text-gray-300">{metric.metric_name.replace("_", " ").toUpperCase()}</p>
        <span className={`rounded-full px-2 py-1 text-xs font-semibold ${metric.regression ? "bg-red-400/10 text-red-300" : "bg-emerald-400/10 text-emerald-300"}`}>
          {metric.regression ? "Regression" : "Normal"}
        </span>
      </div>
      <div className="mt-3 flex items-center justify-between text-xs text-gray-500">
        <span>Baseline: {metric.baseline_value.toFixed(3)}</span>
        <span>Candidate: {metric.candidate_value.toFixed(3)}</span>
        <span className={metric.regression ? "text-red-300" : "text-emerald-300"}>
          {metric.difference > 0 ? "+" : ""}
          {metric.difference.toFixed(3)}
        </span>
      </div>
    </div>
  );
}

function InfoRow({ label, value }: any) {
  return (
    <div className="flex justify-between">
      <span className="text-xs text-gray-500">{label}</span>
      <span className="text-sm font-medium text-gray-300">{value}</span>
    </div>
  );
}

function MiniMetric({ label, value, negative = false }: any) {
  return (
    <div>
      <p className="text-[10px] uppercase tracking-widest text-gray-600">{label}</p>
      <p className={`mt-1 text-lg font-bold ${negative && value < 0 ? "text-red-300" : "text-white"}`}>{value.toFixed(2)}</p>
    </div>
  );
}
