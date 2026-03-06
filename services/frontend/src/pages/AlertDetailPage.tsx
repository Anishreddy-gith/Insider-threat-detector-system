/**
 * Alert Investigation page – full XAI report, evidence panel, analyst actions.
 *
 * Sections:
 *   1. Alert header (title, entity, severity badge, status badge)
 *   2. XAI Report (executive summary, model attribution bar, SHAP waterfall,
 *      baseline comparison, peer-group z-score, investigation playbook)
 *   3. Evidence panel (timeline of raw events that triggered the alert)
 *   4. Analyst actions (Confirm TP / False Positive / Escalate + notes)
 *   5. Audit trail (lifecycle history of this alert)
 */

import { useState } from "react";
import { useParams, Link } from "react-router-dom";
import { format, formatDistanceToNow } from "date-fns";
import {
  ArrowLeft,
  ShieldCheck,
  ShieldX,
  AlertTriangle,
  FileText,
  Clock,
  ChevronDown,
  ChevronUp,
  Send,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

import { useAlert, useUpdateAlertStatus, useSubmitFeedback } from "@/hooks/useAlerts";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  Badge,
  Button,
  Spinner,
} from "@/components/ui";
import { cn, riskLabel } from "@/lib/utils";
import { useAuthStore } from "@/store/authStore";
import type { RiskLevel, FeedbackVerdict, SHAPFeature } from "@/lib/types";

/* ── SHAP colour logic ──────────────────────────────────────────────── */

function shapColor(value: number): string {
  return value > 0 ? "#ef4444" : "#22c55e";
}

/* ── Main Component ─────────────────────────────────────────────────── */

export default function AlertDetailPage() {
  const { id } = useParams<{ id: string }>();
  const { data: alert, isLoading } = useAlert(id);
  const updateStatus = useUpdateAlertStatus();
  const submitFeedback = useSubmitFeedback();
  const role = useAuthStore((s) => s.user?.role);

  const [notes, setNotes] = useState("");
  const [expandedXAI, setExpandedXAI] = useState(true);
  const [expandedEvidence, setExpandedEvidence] = useState(true);
  const [expandedAudit, setExpandedAudit] = useState(false);

  const canAct = role === "admin" || role === "analyst";

  async function handleStatusChange(newStatus: string) {
    if (!alert) return;
    updateStatus.mutate({ alertId: alert.id, status: newStatus });
  }

  async function handleFeedback(verdict: FeedbackVerdict) {
    if (!alert) return;
    submitFeedback.mutate({
      alert_id: alert.id,
      verdict,
      analyst_notes: notes,
    });
    setNotes("");
  }

  if (isLoading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <Spinner className="h-8 w-8" />
      </div>
    );
  }

  if (!alert) {
    return <p className="text-muted-foreground">Alert not found.</p>;
  }

  const xai = alert.xai_report;

  // SHAP feature data for bar chart
  const shapData: SHAPFeature[] = xai?.shap_features ?? [];
  const sortedShap = [...shapData].sort(
    (a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value)
  );

  return (
    <div className="space-y-6">
      {/* ── Back link ──────────────────────────────────────── */}
      <Link
        to="/alerts"
        className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
      >
        <ArrowLeft className="h-4 w-4" /> Back to alerts
      </Link>

      {/* ── Header ─────────────────────────────────────────── */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold">{alert.title}</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Entity:{" "}
            <Link
              to={`/entities/${alert.entity_id}`}
              className="text-primary hover:underline"
            >
              {alert.entity_name}
            </Link>{" "}
            · {format(new Date(alert.created_at), "PPpp")}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant={alert.severity as RiskLevel}>{alert.severity}</Badge>
          <Badge
            variant={
              alert.status === "NEW" || alert.status === "INVESTIGATING"
                ? "destructive"
                : "secondary"
            }
          >
            {alert.status}
          </Badge>
        </div>
      </div>

      {/* ── Two-column layout ──────────────────────────────── */}
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
        {/* ── Left: XAI + Evidence (2 cols) ──────────────── */}
        <div className="space-y-6 xl:col-span-2">
          {/* Description + Risk Score */}
          <Card>
            <CardContent className="space-y-4 p-6">
              <p>{alert.description}</p>
              <div className="flex items-baseline gap-3">
                <span className="text-4xl font-bold tabular-nums">
                  {alert.risk_score.toFixed(1)}
                </span>
                <span className="text-sm text-muted-foreground">
                  / 100 ({riskLabel(alert.risk_score)})
                </span>
              </div>

              {/* Context Factors */}
              {alert.context_factors && alert.context_factors.length > 0 && (
                <div className="rounded-md border p-3">
                  <p className="mb-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                    Context Factors
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {alert.context_factors.map((cf) => (
                      <Badge key={cf.factor} variant="outline">
                        {cf.factor} ×{cf.multiplier.toFixed(1)}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* ── XAI Report ─────────────────────────────────── */}
          {xai && (
            <Card>
              <CardHeader
                className="cursor-pointer pb-2"
                onClick={() => setExpandedXAI(!expandedXAI)}
              >
                <CardTitle className="flex items-center justify-between text-base">
                  <span className="flex items-center gap-2">
                    <FileText className="h-4 w-4" />
                    AI Explanation Report
                  </span>
                  {expandedXAI ? (
                    <ChevronUp className="h-4 w-4" />
                  ) : (
                    <ChevronDown className="h-4 w-4" />
                  )}
                </CardTitle>
              </CardHeader>
              {expandedXAI && (
                <CardContent className="space-y-6">
                  {/* Executive Summary */}
                  <div>
                    <p className="mb-1 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                      Executive Summary
                    </p>
                    <p className="text-sm leading-relaxed">
                      {xai.executive_summary}
                    </p>
                  </div>

                  {/* Model Attribution */}
                  <div>
                    <p className="mb-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                      Model Attribution
                    </p>
                    <div className="space-y-2">
                      {xai.model_attribution.map((m) => (
                        <div key={m.detector} className="space-y-1">
                          <div className="flex items-center justify-between text-sm">
                            <span className="font-medium">{m.detector}</span>
                            <span className="font-mono text-xs text-muted-foreground">
                              {m.contribution_pct.toFixed(1)}% ·{" "}
                              {m.anomaly_type}
                            </span>
                          </div>
                          <div className="h-2 overflow-hidden rounded-full bg-muted">
                            <div
                              className="h-full rounded-full bg-primary transition-all"
                              style={{
                                width: `${Math.min(m.contribution_pct, 100)}%`,
                              }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* SHAP Waterfall Chart */}
                  {sortedShap.length > 0 && (
                    <div>
                      <p className="mb-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                        Feature Importance (SHAP Values)
                      </p>
                      <ResponsiveContainer width="100%" height={Math.max(200, sortedShap.length * 32)}>
                        <BarChart
                          data={sortedShap}
                          layout="vertical"
                          margin={{ left: 120, right: 20, top: 5, bottom: 5 }}
                        >
                          <CartesianGrid
                            strokeDasharray="3 3"
                            stroke="hsl(var(--border))"
                            horizontal={false}
                          />
                          <XAxis
                            type="number"
                            tick={{
                              fontSize: 11,
                              fill: "hsl(var(--muted-foreground))",
                            }}
                          />
                          <YAxis
                            type="category"
                            dataKey="feature"
                            tick={{
                              fontSize: 11,
                              fill: "hsl(var(--muted-foreground))",
                            }}
                            width={115}
                          />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: "hsl(var(--card))",
                              border: "1px solid hsl(var(--border))",
                              borderRadius: "8px",
                            }}
                            formatter={(value: number) => [
                              value.toFixed(4),
                              "SHAP",
                            ]}
                          />
                          <Bar dataKey="shap_value" radius={[0, 4, 4, 0]}>
                            {sortedShap.map((entry, i) => (
                              <Cell
                                key={i}
                                fill={shapColor(entry.shap_value)}
                                fillOpacity={0.7}
                              />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  )}

                  {/* Baseline Comparison */}
                  <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
                    <div className="rounded-md border p-3 text-center">
                      <p className="text-xs text-muted-foreground">Current</p>
                      <p className="text-xl font-bold">
                        {xai.baseline_comparison.current_score.toFixed(1)}
                      </p>
                    </div>
                    <div className="rounded-md border p-3 text-center">
                      <p className="text-xs text-muted-foreground">Baseline</p>
                      <p className="text-xl font-bold">
                        {xai.baseline_comparison.baseline_score.toFixed(1)}
                      </p>
                    </div>
                    <div className="rounded-md border p-3 text-center">
                      <p className="text-xs text-muted-foreground">Deviation</p>
                      <p
                        className={cn(
                          "text-xl font-bold",
                          xai.baseline_comparison.deviation_pct > 0
                            ? "text-risk-critical"
                            : "text-risk-low"
                        )}
                      >
                        {xai.baseline_comparison.deviation_pct > 0 ? "+" : ""}
                        {xai.baseline_comparison.deviation_pct.toFixed(1)}%
                      </p>
                    </div>
                  </div>

                  {/* Peer Group Analysis */}
                  <div className="rounded-md border p-4">
                    <p className="mb-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                      Peer Group Analysis
                    </p>
                    <div className="grid grid-cols-2 gap-4 text-sm sm:grid-cols-4">
                      <div>
                        <p className="text-muted-foreground">Group</p>
                        <p className="font-medium">
                          {xai.peer_group_analysis.peer_group}
                        </p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Peers</p>
                        <p className="font-medium">
                          {xai.peer_group_analysis.peer_count}
                        </p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Z-Score</p>
                        <p
                          className={cn(
                            "font-bold",
                            Math.abs(xai.peer_group_analysis.user_zscore) > 2
                              ? "text-risk-critical"
                              : "text-foreground"
                          )}
                        >
                          {xai.peer_group_analysis.user_zscore.toFixed(2)}σ
                        </p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Peer Mean</p>
                        <p className="font-medium">
                          {xai.peer_group_analysis.peer_mean.toFixed(1)}
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Investigation Playbook */}
                  <div>
                    <p className="mb-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                      Investigation Playbook
                    </p>
                    <ol className="list-inside list-decimal space-y-1 text-sm">
                      {xai.investigation_playbook.map((step, i) => (
                        <li key={i} className="leading-relaxed">
                          {step}
                        </li>
                      ))}
                    </ol>
                  </div>
                </CardContent>
              )}
            </Card>
          )}

          {/* ── Evidence Panel ──────────────────────────────── */}
          {alert.evidence && alert.evidence.length > 0 && (
            <Card>
              <CardHeader
                className="cursor-pointer pb-2"
                onClick={() => setExpandedEvidence(!expandedEvidence)}
              >
                <CardTitle className="flex items-center justify-between text-base">
                  <span className="flex items-center gap-2">
                    <Clock className="h-4 w-4" />
                    Evidence Timeline ({alert.evidence.length})
                  </span>
                  {expandedEvidence ? (
                    <ChevronUp className="h-4 w-4" />
                  ) : (
                    <ChevronDown className="h-4 w-4" />
                  )}
                </CardTitle>
              </CardHeader>
              {expandedEvidence && (
                <CardContent>
                  <div className="relative space-y-0 border-l-2 border-muted pl-6">
                    {alert.evidence.map((ev, i) => (
                      <div key={i} className="relative pb-6 last:pb-0">
                        {/* Timeline dot */}
                        <div className="absolute -left-[31px] top-1 h-3 w-3 rounded-full border-2 border-background bg-primary" />
                        <div className="space-y-1">
                          <div className="flex items-center gap-2 text-xs text-muted-foreground">
                            <Badge variant="outline" className="text-[10px]">
                              {ev.type}
                            </Badge>
                            <span>
                              {format(new Date(ev.timestamp), "PPpp")}
                            </span>
                            <span className="font-mono">
                              +{ev.risk_contribution.toFixed(2)}
                            </span>
                          </div>
                          <p className="text-sm">{ev.description}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              )}
            </Card>
          )}
        </div>

        {/* ── Right sidebar ──────────────────────────────── */}
        <div className="space-y-6">
          {/* Status Actions */}
          {canAct && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">Actions</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {(alert.status === "NEW") && (
                  <Button
                    className="w-full"
                    onClick={() => handleStatusChange("INVESTIGATING")}
                    disabled={updateStatus.isPending}
                  >
                    Start Investigation
                  </Button>
                )}
                {alert.status === "INVESTIGATING" && (
                  <>
                    <Button
                      className="w-full"
                      onClick={() => handleStatusChange("RESOLVED")}
                      disabled={updateStatus.isPending}
                    >
                      <ShieldCheck className="mr-2 h-4 w-4" />
                      Mark Resolved
                    </Button>
                    <Button
                      variant="destructive"
                      className="w-full"
                      onClick={() => handleStatusChange("ESCALATED")}
                      disabled={updateStatus.isPending}
                    >
                      <AlertTriangle className="mr-2 h-4 w-4" />
                      Escalate
                    </Button>
                  </>
                )}

                {/* Feedback */}
                <div className="border-t pt-3">
                  <p className="mb-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                    Analyst Feedback
                  </p>
                  <textarea
                    value={notes}
                    onChange={(e) => setNotes(e.target.value)}
                    placeholder="Investigation notes…"
                    className="mb-2 w-full rounded-md border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                    rows={3}
                  />
                  <div className="grid grid-cols-2 gap-2">
                    <Button
                      size="sm"
                      onClick={() => handleFeedback("TRUE_POSITIVE")}
                      disabled={submitFeedback.isPending}
                    >
                      <ShieldCheck className="mr-1 h-3 w-3" />
                      Confirm TP
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => handleFeedback("FALSE_POSITIVE")}
                      disabled={submitFeedback.isPending}
                    >
                      <ShieldX className="mr-1 h-3 w-3" />
                      False Positive
                    </Button>
                    <Button
                      size="sm"
                      variant="destructive"
                      className="col-span-2"
                      onClick={() => handleFeedback("ESCALATE")}
                      disabled={submitFeedback.isPending}
                    >
                      <Send className="mr-1 h-3 w-3" />
                      Escalate
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Recommended Actions */}
          {alert.recommended_actions && alert.recommended_actions.length > 0 && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">Recommended Actions</CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="list-inside list-disc space-y-1 text-sm">
                  {alert.recommended_actions.map((action, i) => (
                    <li key={i}>{action}</li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          )}

          {/* Audit Trail */}
          {alert.audit_trail && alert.audit_trail.length > 0 && (
            <Card>
              <CardHeader
                className="cursor-pointer pb-2"
                onClick={() => setExpandedAudit(!expandedAudit)}
              >
                <CardTitle className="flex items-center justify-between text-base">
                  Audit Trail ({alert.audit_trail.length})
                  {expandedAudit ? (
                    <ChevronUp className="h-4 w-4" />
                  ) : (
                    <ChevronDown className="h-4 w-4" />
                  )}
                </CardTitle>
              </CardHeader>
              {expandedAudit && (
                <CardContent>
                  <div className="space-y-3">
                    {alert.audit_trail.map((entry, i) => (
                      <div key={i} className="text-xs">
                        <div className="flex items-center justify-between">
                          <Badge variant="outline" className="text-[10px]">
                            {entry.action}
                          </Badge>
                          <span className="text-muted-foreground">
                            {formatDistanceToNow(new Date(entry.timestamp), {
                              addSuffix: true,
                            })}
                          </span>
                        </div>
                        <p className="mt-0.5 text-muted-foreground">
                          {entry.actor}: {entry.details}
                        </p>
                      </div>
                    ))}
                  </div>
                </CardContent>
              )}
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
