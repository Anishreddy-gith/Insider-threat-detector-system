/**
 * Risk Dashboard – SOC analyst command centre.
 *
 * Layout:
 *   Row 1: 4 KPI stat cards (total alerts, critical entities, avg risk, monitored)
 *   Row 2: Department heatmap (TreeMap) | Live alert feed (WebSocket)
 *   Row 3: Risk distribution donut | Alert trend area chart
 *   Row 4: Top risky entities table
 *
 * Data refreshes via React Query (10s) + WebSocket push invalidation.
 */

import { useMemo, useCallback } from "react";
import { Link } from "react-router-dom";
import {
  PieChart,
  Pie,
  Cell,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  Treemap,
} from "recharts";
import {
  AlertTriangle,
  ShieldAlert,
  TrendingUp,
  Users,
  Radio,
} from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import { useQuery } from "@tanstack/react-query";

import { fetchDashboard } from "@/lib/services";
import { useWebSocket } from "@/hooks/useWebSocket";
import { useLiveFeedStore } from "@/store/liveFeedStore";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  Badge,
  Spinner,
} from "@/components/ui";
import { cn, riskLabel, riskBgColor } from "@/lib/utils";
import type {
  RiskLevel,
  LiveFeedItem,
  Alert as AlertT,
  RiskScore,
} from "@/lib/types";

/* ── Constants ──────────────────────────────────────────────────────── */

const PIE_COLORS: Record<string, string> = {
  low: "#22c55e",
  medium: "#eab308",
  high: "#f97316",
  critical: "#ef4444",
};

const RISK_FILL: Record<string, string> = {
  low: "#22c55e",
  medium: "#eab308",
  high: "#f97316",
  critical: "#ef4444",
};

/* ── Custom Treemap content renderer ────────────────────────────────── */

interface TreemapContentProps {
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  name?: string;
  avg_risk?: number;
  critical_count?: number;
}

function TreemapContent(props: TreemapContentProps) {
  const {
    x = 0,
    y = 0,
    width = 0,
    height = 0,
    name,
    avg_risk = 0,
    critical_count = 0,
  } = props;

  if (width < 40 || height < 30) return null;

  const level = riskLabel(avg_risk).toLowerCase();
  const fill = RISK_FILL[level] ?? "#64748b";

  return (
    <g>
      <rect
        x={x}
        y={y}
        width={width}
        height={height}
        rx={4}
        fill={fill}
        fillOpacity={0.2}
        stroke={fill}
        strokeWidth={1.5}
        className="transition-opacity hover:fill-opacity-40"
      />
      <text
        x={x + width / 2}
        y={y + height / 2 - 8}
        textAnchor="middle"
        fill="currentColor"
        className="text-xs font-medium fill-foreground"
        fontSize={width > 100 ? 13 : 11}
      >
        {name}
      </text>
      <text
        x={x + width / 2}
        y={y + height / 2 + 10}
        textAnchor="middle"
        className="text-[10px] fill-muted-foreground"
        fontSize={10}
      >
        {avg_risk.toFixed(0)} avg · {critical_count} crit
      </text>
    </g>
  );
}

/* ── Main Component ─────────────────────────────────────────────────── */

export default function DashboardPage() {
  const pushFeed = useLiveFeedStore((s) => s.push);
  const feedItems = useLiveFeedStore((s) => s.items);

  // Build live-feed items from WebSocket events
  const handleAlertCreated = useCallback(
    (alert: AlertT) => {
      const item: LiveFeedItem = {
        id: `alert-${alert.id}-${Date.now()}`,
        type: "alert",
        entity_id: alert.entity_id,
        entity_name: alert.entity_name,
        message: alert.title,
        severity: alert.severity,
        timestamp: alert.created_at,
      };
      pushFeed(item);
    },
    [pushFeed]
  );

  const handleRiskUpdate = useCallback(
    (score: RiskScore) => {
      const item: LiveFeedItem = {
        id: `risk-${score.entity_id}-${Date.now()}`,
        type: "risk_change",
        entity_id: score.entity_id,
        entity_name: score.username,
        message: `Risk score changed to ${score.risk_score.toFixed(1)}`,
        severity: score.risk_level,
        timestamp: score.timestamp,
      };
      pushFeed(item);
    },
    [pushFeed]
  );

  // Connect WebSocket
  useWebSocket({
    onAlertCreated: handleAlertCreated,
    onRiskUpdate: handleRiskUpdate,
    onLiveFeed: (item) => pushFeed(item),
  });

  // Fetch dashboard data
  const { data: metrics, isLoading } = useQuery({
    queryKey: ["dashboard"],
    queryFn: fetchDashboard,
    refetchInterval: 10_000,
    staleTime: 5_000,
  });

  // Treemap data
  const treemapData = useMemo(() => {
    if (!metrics?.department_heatmap) return [];
    return metrics.department_heatmap.map((d) => ({
      name: d.department,
      size: d.entity_count,
      avg_risk: d.avg_risk,
      critical_count: d.critical_count,
    }));
  }, [metrics?.department_heatmap]);

  if (isLoading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <Spinner className="h-8 w-8" />
      </div>
    );
  }

  if (!metrics) {
    return <p className="text-muted-foreground">Failed to load dashboard.</p>;
  }

  const kpis = [
    {
      label: "Total Alerts",
      value: metrics.total_alerts,
      icon: AlertTriangle,
      color: "text-risk-high",
      bg: "bg-risk-high/10",
    },
    {
      label: "Critical Entities",
      value: metrics.critical_entities,
      icon: ShieldAlert,
      color: "text-risk-critical",
      bg: "bg-risk-critical/10",
    },
    {
      label: "Avg Risk Score",
      value: metrics.average_risk_score.toFixed(1),
      icon: TrendingUp,
      color: "text-primary",
      bg: "bg-primary/10",
    },
    {
      label: "Monitored Users",
      value: metrics.active_entities,
      icon: Users,
      color: "text-muted-foreground",
      bg: "bg-muted",
    },
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Risk Dashboard</h1>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <Radio className="h-3 w-3 animate-pulse text-green-500" />
          Live
        </div>
      </div>

      {/* ── KPI Cards ───────────────────────────────────────── */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {kpis.map(({ label, value, icon: Icon, color, bg }) => (
          <Card key={label} className="overflow-hidden">
            <CardContent className="flex items-center gap-4 p-5">
              <div className={cn("rounded-lg p-2.5", bg)}>
                <Icon className={cn("h-6 w-6", color)} />
              </div>
              <div>
                <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                  {label}
                </p>
                <p className="text-2xl font-bold tabular-nums">{value}</p>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* ── Row 2: Heatmap + Live Feed ──────────────────────── */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Department Heatmap */}
        <Card className="lg:col-span-2">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">
              Department Risk Heatmap
            </CardTitle>
          </CardHeader>
          <CardContent>
            {treemapData.length > 0 ? (
              <ResponsiveContainer width="100%" height={280}>
                <Treemap
                  data={treemapData}
                  dataKey="size"
                  aspectRatio={4 / 3}
                  stroke="hsl(var(--border))"
                  content={<TreemapContent />}
                />
              </ResponsiveContainer>
            ) : (
              <div className="flex h-[280px] items-center justify-center text-sm text-muted-foreground">
                No department data available
              </div>
            )}
          </CardContent>
        </Card>

        {/* Live Alert Feed */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-base">
              <Radio className="h-3.5 w-3.5 animate-pulse text-green-500" />
              Live Feed
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="max-h-[260px] space-y-2 overflow-y-auto pr-1">
              {feedItems.length === 0 ? (
                <p className="py-8 text-center text-xs text-muted-foreground">
                  Waiting for real-time events…
                </p>
              ) : (
                feedItems.map((item) => (
                  <Link
                    key={item.id}
                    to={
                      item.type === "alert"
                        ? `/alerts`
                        : `/entities/${item.entity_id}`
                    }
                    className="flex items-start gap-2 rounded-md border px-3 py-2 text-xs transition-colors hover:bg-accent"
                  >
                    <Badge
                      variant={item.severity as RiskLevel}
                      className="mt-0.5 shrink-0"
                    >
                      {item.severity}
                    </Badge>
                    <div className="min-w-0 flex-1">
                      <p className="truncate font-medium">{item.message}</p>
                      <p className="text-muted-foreground">
                        {item.entity_name} ·{" "}
                        {formatDistanceToNow(new Date(item.timestamp), {
                          addSuffix: true,
                        })}
                      </p>
                    </div>
                  </Link>
                ))
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* ── Row 3: Charts ───────────────────────────────────── */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Risk Distribution Donut */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Risk Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie
                  data={metrics.risk_distribution}
                  dataKey="count"
                  nameKey="level"
                  cx="50%"
                  cy="50%"
                  outerRadius={95}
                  innerRadius={55}
                  paddingAngle={3}
                  label={({
                    level,
                    count,
                  }: {
                    level: string;
                    count: number;
                  }) => `${level}: ${count}`}
                >
                  {metrics.risk_distribution.map((entry) => (
                    <Cell
                      key={entry.level}
                      fill={PIE_COLORS[entry.level] ?? "#8884d8"}
                    />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: "hsl(var(--card))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "8px",
                  }}
                />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Alert Trend */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">
              Alert Trend (7 days)
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={260}>
              <AreaChart data={metrics.alert_trend}>
                <defs>
                  <linearGradient id="alertGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop
                      offset="5%"
                      stopColor="hsl(221.2,83.2%,53.3%)"
                      stopOpacity={0.3}
                    />
                    <stop
                      offset="95%"
                      stopColor="hsl(221.2,83.2%,53.3%)"
                      stopOpacity={0}
                    />
                  </linearGradient>
                </defs>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="hsl(var(--border))"
                />
                <XAxis
                  dataKey="date"
                  tick={{
                    fontSize: 11,
                    fill: "hsl(var(--muted-foreground))",
                  }}
                />
                <YAxis
                  allowDecimals={false}
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
                <Area
                  type="monotone"
                  dataKey="count"
                  stroke="hsl(221.2,83.2%,53.3%)"
                  fill="url(#alertGrad)"
                  strokeWidth={2}
                />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* ── Row 4: Top Risky Entities ───────────────────────── */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Top Risky Entities</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b text-left text-xs uppercase tracking-wider text-muted-foreground">
                  <th className="pb-3 font-medium">Entity</th>
                  <th className="pb-3 font-medium">Department</th>
                  <th className="pb-3 font-medium">Risk Score</th>
                  <th className="pb-3 font-medium">Level</th>
                  <th className="pb-3 font-medium">Risk Bar</th>
                </tr>
              </thead>
              <tbody>
                {metrics.top_entities.map((e) => (
                  <tr
                    key={e.entity_id}
                    className="border-b last:border-0 transition-colors hover:bg-muted/50"
                  >
                    <td className="py-2.5">
                      <Link
                        to={`/entities/${e.entity_id}`}
                        className="font-medium text-primary hover:underline"
                      >
                        {e.username}
                      </Link>
                    </td>
                    <td className="py-2.5 text-muted-foreground">
                      {e.department}
                    </td>
                    <td className="py-2.5 font-mono tabular-nums">
                      {e.risk_score.toFixed(1)}
                    </td>
                    <td className="py-2.5">
                      <Badge
                        variant={
                          riskLabel(e.risk_score).toLowerCase() as RiskLevel
                        }
                      >
                        {riskLabel(e.risk_score)}
                      </Badge>
                    </td>
                    <td className="py-2.5 w-32">
                      <div className="h-2 overflow-hidden rounded-full bg-muted">
                        <div
                          className={cn(
                            "h-full rounded-full transition-all",
                            riskBgColor(e.risk_score)
                          )}
                          style={{
                            width: `${Math.min(e.risk_score, 100)}%`,
                          }}
                        />
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
