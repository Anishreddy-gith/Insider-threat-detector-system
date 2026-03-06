/**
 * User Risk Profile page – deep-dive into one entity.
 *
 * Sections:
 *   1. Identity header (username, department, title, hire date, risk gauge)
 *   2. 30-day risk score trend (LineChart with reference lines)
 *   3. Behaviour radar chart (8 axes)
 *   4. Peer comparison bar chart (user vs peer mean vs p95)
 *   5. Context factors
 *   6. XAI report (reused section from AlertDetail)
 *   7. Recent alerts list
 */

import { useParams, Link } from "react-router-dom";
import { format, formatDistanceToNow } from "date-fns";
import { ArrowLeft, User, Building2, Calendar, Briefcase } from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  BarChart,
  Bar,
  Legend,
} from "recharts";

import { useUserProfile } from "@/hooks/useUserProfile";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  Badge,
  Spinner,
} from "@/components/ui";
import { cn, riskLabel } from "@/lib/utils";
import type { RiskLevel } from "@/lib/types";

/* ── Behaviour features → radar data ───────────────────────────────── */

const RADAR_LABELS: Record<string, string> = {
  login_frequency: "Login Freq",
  after_hours_ratio: "After Hours",
  file_access_volume: "File Access",
  email_external_ratio: "Ext Email",
  usb_usage: "USB Usage",
  network_anomaly: "Net Anomaly",
  badge_anomaly: "Badge Anomaly",
  data_exfiltration_score: "Exfil Score",
};

export default function EntityDetailPage() {
  const { id } = useParams<{ id: string }>();
  const { data: profile, isLoading } = useUserProfile(id);

  if (isLoading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <Spinner className="h-8 w-8" />
      </div>
    );
  }

  if (!profile) {
    return <p className="text-muted-foreground">Entity not found.</p>;
  }

  // Radar chart data from behaviour features
  const radarData = Object.entries(profile.behavior_features).map(
    ([key, value]) => ({
      feature: RADAR_LABELS[key] ?? key,
      value: Math.min(value * 100, 100), // normalize to 0-100
      fullMark: 100,
    })
  );

  // Peer comparison bar data
  const peerBarData =
    profile.peer_comparison?.metrics?.map((m) => ({
      feature: m.feature,
      user: m.user_value,
      peer_mean: m.peer_mean,
      peer_p95: m.peer_p95,
    })) ?? [];

  return (
    <div className="space-y-6">
      {/* ── Back link ──────────────────────────────────────── */}
      <Link
        to="/entities"
        className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
      >
        <ArrowLeft className="h-4 w-4" /> Back to entities
      </Link>

      {/* ── Identity header ────────────────────────────────── */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="space-y-1">
          <h1 className="text-2xl font-bold">{profile.username}</h1>
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-sm text-muted-foreground">
            <span className="flex items-center gap-1">
              <User className="h-3.5 w-3.5" /> {profile.title || "N/A"}
            </span>
            <span className="flex items-center gap-1">
              <Building2 className="h-3.5 w-3.5" /> {profile.department}
            </span>
            <span className="flex items-center gap-1">
              <Briefcase className="h-3.5 w-3.5" /> Mgr: {profile.manager || "N/A"}
            </span>
            <span className="flex items-center gap-1">
              <Calendar className="h-3.5 w-3.5" />{" "}
              Hired {profile.hire_date ? format(new Date(profile.hire_date), "PP") : "N/A"}
            </span>
          </div>
        </div>

        {/* Risk gauge */}
        <div className="flex items-center gap-3">
          <div className="text-right">
            <p className="text-3xl font-bold tabular-nums">
              {profile.risk_score.toFixed(1)}
            </p>
            <Badge
              variant={
                riskLabel(profile.risk_score).toLowerCase() as RiskLevel
              }
            >
              {riskLabel(profile.risk_score)}
            </Badge>
          </div>
          <div className="h-16 w-16">
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
                className={cn(
                  "transition-all duration-700",
                  riskLabel(profile.risk_score).toLowerCase() === "critical"
                    ? "stroke-risk-critical"
                    : riskLabel(profile.risk_score).toLowerCase() === "high"
                    ? "stroke-risk-high"
                    : riskLabel(profile.risk_score).toLowerCase() === "medium"
                    ? "stroke-risk-medium"
                    : "stroke-risk-low"
                )}
                strokeWidth="3"
                strokeDasharray={`${profile.risk_score} ${100 - profile.risk_score}`}
                strokeLinecap="round"
              />
            </svg>
          </div>
        </div>
      </div>

      {/* ── Row 1: 30-day trend ────────────────────────────── */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">
            Risk Score Trend (30 days)
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={profile.risk_history}>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="hsl(var(--border))"
              />
              <XAxis
                dataKey="timestamp"
                tickFormatter={(v) => format(new Date(v), "MM/dd")}
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
              />
              <Tooltip
                labelFormatter={(v) =>
                  format(new Date(v as string), "PPpp")
                }
                contentStyle={{
                  backgroundColor: "hsl(var(--card))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "8px",
                }}
              />
              <ReferenceLine
                y={75}
                stroke="#ef4444"
                strokeDasharray="3 3"
                label={{ value: "Critical", fill: "#ef4444", fontSize: 10 }}
              />
              <ReferenceLine
                y={50}
                stroke="#f97316"
                strokeDasharray="3 3"
                label={{ value: "High", fill: "#f97316", fontSize: 10 }}
              />
              <ReferenceLine
                y={25}
                stroke="#eab308"
                strokeDasharray="3 3"
                label={{ value: "Medium", fill: "#eab308", fontSize: 10 }}
              />
              <Line
                type="monotone"
                dataKey="score"
                stroke="hsl(221.2,83.2%,53.3%)"
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* ── Row 2: Radar + Peer Comparison ─────────────────── */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Behaviour Radar */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Behaviour Profile</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={320}>
              <RadarChart cx="50%" cy="50%" outerRadius="70%" data={radarData}>
                <PolarGrid stroke="hsl(var(--border))" />
                <PolarAngleAxis
                  dataKey="feature"
                  tick={{
                    fontSize: 11,
                    fill: "hsl(var(--muted-foreground))",
                  }}
                />
                <PolarRadiusAxis
                  angle={90}
                  domain={[0, 100]}
                  tick={{ fontSize: 9, fill: "hsl(var(--muted-foreground))" }}
                />
                <Radar
                  name="User"
                  dataKey="value"
                  stroke="hsl(221.2,83.2%,53.3%)"
                  fill="hsl(221.2,83.2%,53.3%)"
                  fillOpacity={0.25}
                  strokeWidth={2}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "hsl(var(--card))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "8px",
                  }}
                />
              </RadarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Peer Comparison */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">
              Peer Comparison
              {profile.peer_comparison && (
                <span className="ml-2 text-xs font-normal text-muted-foreground">
                  ({profile.peer_comparison.peer_group} · {profile.peer_comparison.peer_count} peers)
                </span>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {peerBarData.length > 0 ? (
              <ResponsiveContainer width="100%" height={320}>
                <BarChart
                  data={peerBarData}
                  margin={{ top: 5, right: 20, left: 20, bottom: 5 }}
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="hsl(var(--border))"
                  />
                  <XAxis
                    dataKey="feature"
                    tick={{
                      fontSize: 10,
                      fill: "hsl(var(--muted-foreground))",
                    }}
                    angle={-25}
                    textAnchor="end"
                    height={60}
                  />
                  <YAxis
                    tick={{
                      fontSize: 11,
                      fill: "hsl(var(--muted-foreground))",
                    }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "hsl(var(--card))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "8px",
                    }}
                  />
                  <Legend />
                  <Bar
                    dataKey="user"
                    name="User"
                    fill="hsl(221.2,83.2%,53.3%)"
                    radius={[4, 4, 0, 0]}
                  />
                  <Bar
                    dataKey="peer_mean"
                    name="Peer Mean"
                    fill="#64748b"
                    radius={[4, 4, 0, 0]}
                  />
                  <Bar
                    dataKey="peer_p95"
                    name="Peer P95"
                    fill="#f97316"
                    fillOpacity={0.5}
                    radius={[4, 4, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-[320px] items-center justify-center text-sm text-muted-foreground">
                No peer comparison data available
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* ── Context Factors ────────────────────────────────── */}
      {profile.context_factors && profile.context_factors.length > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Context Factors</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {profile.context_factors.map((cf) => (
                <div
                  key={cf.factor}
                  className="flex items-center justify-between rounded-md border p-3"
                >
                  <div>
                    <p className="text-sm font-medium">{cf.factor}</p>
                    <p className="text-xs text-muted-foreground">
                      {cf.description}
                    </p>
                  </div>
                  <Badge
                    variant={cf.multiplier > 1 ? "destructive" : "secondary"}
                  >
                    ×{cf.multiplier.toFixed(1)}
                  </Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* ── XAI Report ─────────────────────────────────────── */}
      {profile.xai_report && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">AI Explanation</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-sm leading-relaxed">
              {profile.xai_report.executive_summary}
            </p>

            {/* Model Attribution */}
            <div>
              <p className="mb-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                Model Attribution
              </p>
              <div className="space-y-2">
                {profile.xai_report.model_attribution.map((m) => (
                  <div key={m.detector} className="space-y-1">
                    <div className="flex items-center justify-between text-sm">
                      <span className="font-medium">{m.detector}</span>
                      <span className="font-mono text-xs text-muted-foreground">
                        {m.contribution_pct.toFixed(1)}%
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

            {/* SHAP Features */}
            {profile.xai_report.shap_features.length > 0 && (
              <div>
                <p className="mb-1 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  Top SHAP Features
                </p>
                <div className="space-y-1">
                  {profile.xai_report.shap_features
                    .sort(
                      (a, b) =>
                        Math.abs(b.shap_value) - Math.abs(a.shap_value)
                    )
                    .slice(0, 8)
                    .map((f) => (
                      <div
                        key={f.feature}
                        className="flex items-center justify-between py-1 text-sm"
                      >
                        <span>{f.feature}</span>
                        <span
                          className={cn(
                            "font-mono text-xs",
                            f.shap_value > 0
                              ? "text-risk-critical"
                              : "text-risk-low"
                          )}
                        >
                          {f.shap_value > 0 ? "+" : ""}
                          {f.shap_value.toFixed(4)}
                        </span>
                      </div>
                    ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* ── Recent Alerts ──────────────────────────────────── */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Recent Alerts</CardTitle>
        </CardHeader>
        <CardContent>
          {(!profile.recent_alerts || profile.recent_alerts.length === 0) ? (
            <p className="py-4 text-center text-sm text-muted-foreground">
              No recent alerts.
            </p>
          ) : (
            <div className="space-y-2">
              {profile.recent_alerts.map((a) => (
                <Link
                  key={a.id}
                  to={`/alerts/${a.id}`}
                  className="flex items-center justify-between rounded-md border px-4 py-3 transition-colors hover:bg-accent"
                >
                  <div>
                    <p className="text-sm font-medium">{a.title}</p>
                    <p className="text-xs text-muted-foreground">
                      {formatDistanceToNow(new Date(a.created_at), {
                        addSuffix: true,
                      })}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant={a.severity as RiskLevel}>
                      {a.severity}
                    </Badge>
                    <Badge
                      variant={
                        a.status === "NEW" || a.status === "INVESTIGATING"
                          ? "destructive"
                          : "secondary"
                      }
                    >
                      {a.status}
                    </Badge>
                  </div>
                </Link>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
