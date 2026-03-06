/**
 * Alerts list page – paginated, filterable alert table.
 *
 * Upgraded to React Query for caching and background refetch.
 */

import { useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { formatDistanceToNow } from "date-fns";

import { useAlerts } from "@/hooks/useAlerts";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  Badge,
  Button,
  Spinner,
} from "@/components/ui";

export default function AlertsPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const page = parseInt(searchParams.get("page") ?? "1", 10);
  const [severityFilter, setSeverityFilter] = useState<string>("all");

  const { data, isLoading } = useAlerts({
    page,
    size: 20,
    severity: severityFilter !== "all" ? severityFilter : undefined,
  });

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Alerts</h1>
        <div className="flex gap-2">
          {["all", "low", "medium", "high", "critical"].map((level) => (
            <Button
              key={level}
              variant={severityFilter === level ? "default" : "outline"}
              size="sm"
              onClick={() => {
                setSeverityFilter(level);
                setSearchParams({ page: "1" });
              }}
            >
              {level === "all"
                ? "All"
                : level.charAt(0).toUpperCase() + level.slice(1)}
            </Button>
          ))}
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>
            {data
              ? `${data.total} alert${data.total !== 1 ? "s" : ""}`
              : "Loading..."}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="flex justify-center py-8">
              <Spinner />
            </div>
          ) : !data || data.items.length === 0 ? (
            <p className="py-8 text-center text-muted-foreground">
              No alerts found.
            </p>
          ) : (
            <>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b text-left text-xs uppercase tracking-wider text-muted-foreground">
                      <th className="pb-3 font-medium">Title</th>
                      <th className="pb-3 font-medium">Entity</th>
                      <th className="pb-3 font-medium">Severity</th>
                      <th className="pb-3 font-medium">Score</th>
                      <th className="pb-3 font-medium">Status</th>
                      <th className="pb-3 font-medium">Time</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.items.map((alert) => (
                      <tr
                        key={alert.id}
                        className="border-b last:border-0 transition-colors hover:bg-muted/50"
                      >
                        <td className="py-2.5">
                          <Link
                            to={`/alerts/${alert.id}`}
                            className="font-medium text-primary hover:underline"
                          >
                            {alert.title}
                          </Link>
                        </td>
                        <td className="py-2.5">
                          <Link
                            to={`/entities/${alert.entity_id}`}
                            className="text-muted-foreground hover:text-foreground"
                          >
                            {alert.entity_name}
                          </Link>
                        </td>
                        <td className="py-2.5">
                          <Badge
                            variant={
                              alert.severity as
                                | "low"
                                | "medium"
                                | "high"
                                | "critical"
                            }
                          >
                            {alert.severity}
                          </Badge>
                        </td>
                        <td className="py-2.5 font-mono tabular-nums">
                          {alert.risk_score.toFixed(1)}
                        </td>
                        <td className="py-2.5">
                          <Badge
                            variant={
                              alert.status === "NEW" ||
                              alert.status === "INVESTIGATING"
                                ? "destructive"
                                : "secondary"
                            }
                          >
                            {alert.status}
                          </Badge>
                        </td>
                        <td className="py-2.5 text-muted-foreground">
                          {formatDistanceToNow(new Date(alert.created_at), {
                            addSuffix: true,
                          })}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Pagination */}
              <div className="mt-4 flex items-center justify-between">
                <p className="text-sm text-muted-foreground">
                  Page {data.page} of {data.pages}
                </p>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={data.page <= 1}
                    onClick={() =>
                      setSearchParams({ page: String(data.page - 1) })
                    }
                  >
                    Previous
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={data.page >= data.pages}
                    onClick={() =>
                      setSearchParams({ page: String(data.page + 1) })
                    }
                  >
                    Next
                  </Button>
                </div>
              </div>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
