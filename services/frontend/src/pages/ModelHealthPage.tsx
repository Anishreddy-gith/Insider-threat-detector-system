/**
 * Model Health page – ML pipeline observability for SOC leads.
 *
 * Sections:
 *   1. Detector cards: precision/recall/F1 gauges per detector
 *   2. Ensemble weights bar chart
 *   3. Privacy budget gauge (ε consumed/total)
 *   4. Federated Learning status card
 *   5. FP/FN analysis per-department table
 */

import { useQuery } from "@tanstack/react-query";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import {
  Activity,
  Wifi,
  Lock,
  AlertTriangle,
  CheckCircle2,
  Clock,
} from "lucide-react";

import { fetchModelHealth } from "@/lib/services";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  Badge,
  Spinner,
} from "@/components/ui";
import { cn } from "@/lib/utils";

/* ── Gauge component ────────────────────────────────────────────────── */

function MetricGauge({
  value,
  label,
  color = "hsl(221.2,83.2%,53.3%)",
}: {
  value: number;
  label: string;
  color?: string;
}) {
  const pct = Math.round(value * 100);
  return (
    <div className="flex flex-col items-center">
      <div className="relative h-16 w-16">
        <svg viewBox="0 0 36 36" className="h-full w-full -rotate-90">
          <circle
            cx="18"
            cy="18"
            r="15.9"
            fill="none"
            stroke="hsl(var(--muted))"
            strokeWidth="3"
          />
          <circle
            cx="18"
            cy="18"
            r="15.9"
            fill="none"
            stroke={color}
            strokeWidth="3"
            strokeDasharray={`${pct} ${100 - pct}`}
            strokeLinecap="round"
            className="transition-all duration-700"
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-xs font-bold tabular-nums">{pct}%</span>
        </div>
      </div>
      <span className="mt-1 text-[10px] uppercase tracking-wider text-muted-foreground">
        {label}
      </span>
    </div>
  );
}

/* ── FL Status badge helper ─────────────────────────────────────────── */

function flStatusBadge(status: string) {
  const map: Record<string, { variant: string; icon: React.ReactNode }> = {
    training: {
      variant: "default",
      icon: <Activity className="mr-1 h-3 w-3" />,
    },
    aggregating: {
      variant: "secondary",
      icon: <Wifi className="mr-1 h-3 w-3" />,
    },
    idle: {
      variant: "outline",
      icon: <Clock className="mr-1 h-3 w-3" />,
    },
    converged: {
      variant: "default",
      icon: <CheckCircle2 className="mr-1 h-3 w-3" />,
    },
  };
  const s = map[status] ?? map.idle;
  return (
    <Badge variant={s.variant as "default"} className="flex items-center">
      {s.icon}
      {status}
    </Badge>
  );
}

/* ── Main Component ─────────────────────────────────────────────────── */

export default function ModelHealthPage() {
  const { data: health, isLoading } = useQuery({
    queryKey: ["modelHealth"],
    queryFn: fetchModelHealth,
    refetchInterval: 30_000,
    staleTime: 15_000,
  });

  if (isLoading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <Spinner className="h-8 w-8" />
      </div>
    );
  }

  if (!health) {
    return (
      <p className="text-muted-foreground">Failed to load model health.</p>
    );
  }

  // Ensemble weights for bar chart
  const weightData = Object.entries(health.ensemble_weights).map(
    ([name, weight]) => ({
      name,
      weight: Number((weight * 100).toFixed(1)),
    })
  );

  // Privacy budget
  const pb = health.privacy_budget;
  const budgetPct =
    pb.total_epsilon > 0
      ? Math.round((pb.consumed_epsilon / pb.total_epsilon) * 100)
      : 0;

  const fl = health.federated_learning;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Model Health</h1>
        <div className="flex items-center gap-2">
          <Activity className="h-4 w-4 text-primary" />
          <span className="text-sm text-muted-foreground">
            {health.detectors.length} detectors active
          </span>
        </div>
      </div>

      {/* ── Detector Cards ──────────────────────────────────── */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {health.detectors.map((d) => (
          <Card key={d.detector}>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-semibold">
                {d.detector.replace(/_/g, " ")}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-around">
                <MetricGauge
                  value={d.precision}
                  label="Precision"
                  color="#22c55e"
                />
                <MetricGauge
                  value={d.recall}
                  label="Recall"
                  color="#3b82f6"
                />
                <MetricGauge
                  value={d.f1_score}
                  label="F1"
                  color="#a855f7"
                />
              </div>
              <div className="mt-4 grid grid-cols-3 gap-2 text-center text-xs">
                <div>
                  <p className="text-muted-foreground">AUC-ROC</p>
                  <p className="font-mono font-bold tabular-nums">
                    {d.auc_roc.toFixed(3)}
                  </p>
                </div>
                <div>
                  <p className="text-muted-foreground">TP / FP</p>
                  <p className="font-mono tabular-nums">
                    <span className="text-risk-low">{d.true_positives}</span>
                    {" / "}
                    <span className="text-risk-critical">
                      {d.false_positives}
                    </span>
                  </p>
                </div>
                <div>
                  <p className="text-muted-foreground">FN</p>
                  <p className="font-mono tabular-nums text-risk-high">
                    {d.false_negatives}
                  </p>
                </div>
              </div>
              <p className="mt-3 text-center text-[10px] text-muted-foreground">
                Trained: {d.last_trained ? new Date(d.last_trained).toLocaleDateString() : "N/A"} ·{" "}
                {d.training_samples.toLocaleString()} samples
              </p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* ── Row 2: Ensemble Weights + Privacy Budget ────────── */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Ensemble Weights */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">
              Ensemble Weights
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart
                data={weightData}
                margin={{ top: 5, right: 20, left: 20, bottom: 5 }}
              >
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="hsl(var(--border))"
                />
                <XAxis
                  dataKey="name"
                  tick={{
                    fontSize: 11,
                    fill: "hsl(var(--muted-foreground))",
                  }}
                />
                <YAxis
                  domain={[0, 100]}
                  tick={{
                    fontSize: 11,
                    fill: "hsl(var(--muted-foreground))",
                  }}
                  label={{
                    value: "%",
                    position: "insideTopLeft",
                    fill: "hsl(var(--muted-foreground))",
                  }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "hsl(var(--card))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "8px",
                  }}
                  formatter={(v: number) => [`${v}%`, "Weight"]}
                />
                <Bar
                  dataKey="weight"
                  fill="hsl(221.2,83.2%,53.3%)"
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Privacy Budget */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-base">
              <Lock className="h-4 w-4" />
              Privacy Budget (Differential Privacy)
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Budget gauge */}
            <div className="flex items-center justify-center">
              <div className="relative h-32 w-32">
                <svg viewBox="0 0 36 36" className="h-full w-full -rotate-90">
                  <circle
                    cx="18"
                    cy="18"
                    r="15.9"
                    fill="none"
                    stroke="hsl(var(--muted))"
                    strokeWidth="3.5"
                  />
                  <circle
                    cx="18"
                    cy="18"
                    r="15.9"
                    fill="none"
                    stroke={
                      budgetPct > 80
                        ? "#ef4444"
                        : budgetPct > 50
                        ? "#f97316"
                        : "#22c55e"
                    }
                    strokeWidth="3.5"
                    strokeDasharray={`${budgetPct} ${100 - budgetPct}`}
                    strokeLinecap="round"
                    className="transition-all duration-700"
                  />
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <span className="text-lg font-bold tabular-nums">
                    {budgetPct}%
                  </span>
                  <span className="text-[9px] text-muted-foreground">
                    consumed
                  </span>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3 text-center text-sm">
              <div className="rounded-md border p-2">
                <p className="text-xs text-muted-foreground">ε consumed</p>
                <p className="font-mono font-bold">
                  {pb.consumed_epsilon.toFixed(2)} / {pb.total_epsilon.toFixed(1)}
                </p>
              </div>
              <div className="rounded-md border p-2">
                <p className="text-xs text-muted-foreground">ε remaining</p>
                <p className="font-mono font-bold text-risk-low">
                  {pb.remaining_epsilon.toFixed(2)}
                </p>
              </div>
              <div className="rounded-md border p-2">
                <p className="text-xs text-muted-foreground">δ consumed</p>
                <p className="font-mono font-bold">
                  {pb.consumed_delta.toExponential(2)}
                </p>
              </div>
              <div className="rounded-md border p-2">
                <p className="text-xs text-muted-foreground">Queries today</p>
                <p className="font-mono font-bold">{pb.queries_today}</p>
              </div>
            </div>

            <p className="text-center text-[10px] text-muted-foreground">
              Budget refreshes:{" "}
              {pb.budget_refresh_at
                ? new Date(pb.budget_refresh_at).toLocaleString()
                : "N/A"}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* ── Row 3: Federated Learning ──────────────────────── */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-base">
            <Wifi className="h-4 w-4" />
            Federated Learning Status
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-6">
            <div className="rounded-md border p-3 text-center">
              <p className="text-xs text-muted-foreground">Status</p>
              <div className="mt-1">{flStatusBadge(fl.status)}</div>
            </div>
            <div className="rounded-md border p-3 text-center">
              <p className="text-xs text-muted-foreground">Round</p>
              <p className="mt-1 text-lg font-bold tabular-nums">
                {fl.current_round} / {fl.total_rounds}
              </p>
            </div>
            <div className="rounded-md border p-3 text-center">
              <p className="text-xs text-muted-foreground">Clients</p>
              <p className="mt-1 text-lg font-bold tabular-nums">
                {fl.participating_clients}
              </p>
            </div>
            <div className="rounded-md border p-3 text-center">
              <p className="text-xs text-muted-foreground">Global Accuracy</p>
              <p className="mt-1 text-lg font-bold tabular-nums">
                {(fl.global_accuracy * 100).toFixed(1)}%
              </p>
            </div>
            <div className="rounded-md border p-3 text-center">
              <p className="text-xs text-muted-foreground">Convergence Δ</p>
              <p className="mt-1 text-lg font-mono font-bold tabular-nums">
                {fl.convergence_delta.toExponential(2)}
              </p>
            </div>
            <div className="rounded-md border p-3 text-center">
              <p className="text-xs text-muted-foreground">Last Aggregation</p>
              <p className="mt-1 text-xs font-medium">
                {fl.last_aggregation
                  ? new Date(fl.last_aggregation).toLocaleString()
                  : "N/A"}
              </p>
            </div>
          </div>

          {/* Progress bar */}
          <div className="mt-4 space-y-1">
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>Training progress</span>
              <span>
                {fl.total_rounds > 0
                  ? Math.round((fl.current_round / fl.total_rounds) * 100)
                  : 0}
                %
              </span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-muted">
              <div
                className="h-full rounded-full bg-primary transition-all"
                style={{
                  width: `${
                    fl.total_rounds > 0
                      ? (fl.current_round / fl.total_rounds) * 100
                      : 0
                  }%`,
                }}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* ── Row 4: FP/FN Analysis per department ───────────── */}
      {health.fp_fn_analysis && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-base">
              <AlertTriangle className="h-4 w-4" />
              FP/FN Analysis by Department
            </CardTitle>
          </CardHeader>
          <CardContent>
            {/* Overall stats */}
            <div className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
              <div className="rounded-md border p-3 text-center">
                <p className="text-xs text-muted-foreground">
                  Overall Precision
                </p>
                <p className="text-lg font-bold tabular-nums">
                  {(health.fp_fn_analysis.overall_precision * 100).toFixed(1)}%
                </p>
              </div>
              <div className="rounded-md border p-3 text-center">
                <p className="text-xs text-muted-foreground">Overall Recall</p>
                <p className="text-lg font-bold tabular-nums">
                  {(health.fp_fn_analysis.overall_recall * 100).toFixed(1)}%
                </p>
              </div>
              <div className="rounded-md border p-3 text-center">
                <p className="text-xs text-muted-foreground">Overall F1</p>
                <p className="text-lg font-bold tabular-nums">
                  {(health.fp_fn_analysis.overall_f1 * 100).toFixed(1)}%
                </p>
              </div>
              <div className="rounded-md border p-3 text-center">
                <p className="text-xs text-muted-foreground">AUC-ROC</p>
                <p className="text-lg font-bold tabular-nums">
                  {health.fp_fn_analysis.overall_auc_roc.toFixed(3)}
                </p>
              </div>
            </div>

            {/* Per-department table */}
            {health.fp_fn_analysis.by_department &&
              health.fp_fn_analysis.by_department.length > 0 && (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b text-left text-xs uppercase tracking-wider text-muted-foreground">
                        <th className="pb-3 font-medium">Department</th>
                        <th className="pb-3 font-medium">Precision</th>
                        <th className="pb-3 font-medium">Recall</th>
                        <th className="pb-3 font-medium">F1</th>
                        <th className="pb-3 font-medium">Alerts</th>
                        <th className="pb-3 font-medium">FP</th>
                      </tr>
                    </thead>
                    <tbody>
                      {health.fp_fn_analysis.by_department.map((dept) => (
                        <tr
                          key={dept.department}
                          className="border-b last:border-0 transition-colors hover:bg-muted/50"
                        >
                          <td className="py-2.5 font-medium">
                            {dept.department}
                          </td>
                          <td className="py-2.5 font-mono tabular-nums">
                            {(dept.precision * 100).toFixed(1)}%
                          </td>
                          <td className="py-2.5 font-mono tabular-nums">
                            {(dept.recall * 100).toFixed(1)}%
                          </td>
                          <td className="py-2.5 font-mono tabular-nums">
                            {(dept.f1_score * 100).toFixed(1)}%
                          </td>
                          <td className="py-2.5 tabular-nums">
                            {dept.alert_count}
                          </td>
                          <td className="py-2.5">
                            <span
                              className={cn(
                                "font-mono tabular-nums",
                                dept.fp_count > 5
                                  ? "text-risk-critical"
                                  : "text-muted-foreground"
                              )}
                            >
                              {dept.fp_count}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
