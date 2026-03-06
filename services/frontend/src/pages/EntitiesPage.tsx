/**
 * Entities list page – monitored users / entities with risk scores.
 *
 * Upgraded to React Query + risk bar visual enhancements.
 */

import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import api from "@/lib/api";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  Badge,
  Button,
  Spinner,
  Input,
} from "@/components/ui";
import { cn, riskLabel, riskBgColor } from "@/lib/utils";

interface Entity {
  id: string;
  username: string;
  department: string;
  risk_score: number;
  alert_count: number;
  last_activity: string;
}

interface PaginatedEntities {
  items: Entity[];
  total: number;
  page: number;
  size: number;
  pages: number;
}

async function fetchEntities(params: {
  page: number;
  size: number;
  search?: string;
}): Promise<PaginatedEntities> {
  const { data } = await api.get("/analytics/entities", { params });
  return data;
}

export default function EntitiesPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const page = parseInt(searchParams.get("page") ?? "1", 10);
  const [search, setSearch] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");

  // Simple debounce
  useEffect(() => {
    const t = setTimeout(() => setDebouncedSearch(search), 300);
    return () => clearTimeout(t);
  }, [search]);

  const { data, isLoading } = useQuery({
    queryKey: ["entities", page, debouncedSearch],
    queryFn: () =>
      fetchEntities({
        page,
        size: 24,
        search: debouncedSearch || undefined,
      }),
    staleTime: 10_000,
  });

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between gap-4">
        <h1 className="text-2xl font-bold">Monitored Entities</h1>
        <Input
          placeholder="Search entities..."
          value={search}
          onChange={(e) => {
            setSearch(e.target.value);
            setSearchParams({ page: "1" });
          }}
          className="max-w-xs"
        />
      </div>

      <Card>
        <CardHeader>
          <CardTitle>
            {data
              ? `${data.total} entit${data.total !== 1 ? "ies" : "y"}`
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
              No entities found.
            </p>
          ) : (
            <>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {data.items.map((entity) => (
                  <Link
                    key={entity.id}
                    to={`/entities/${entity.id}`}
                    className="group rounded-lg border p-4 transition-shadow hover:shadow-md"
                  >
                    <div className="flex items-center justify-between">
                      <p className="font-medium group-hover:text-primary">
                        {entity.username}
                      </p>
                      <Badge
                        variant={
                          riskLabel(entity.risk_score).toLowerCase() as
                            | "low"
                            | "medium"
                            | "high"
                            | "critical"
                        }
                      >
                        {riskLabel(entity.risk_score)}
                      </Badge>
                    </div>
                    <p className="mt-1 text-sm text-muted-foreground">
                      {entity.department}
                    </p>
                    <div className="mt-3">
                      <div className="mb-1 flex justify-between text-xs text-muted-foreground">
                        <span>Risk Score</span>
                        <span className="font-mono tabular-nums">
                          {entity.risk_score.toFixed(1)}
                        </span>
                      </div>
                      <div className="h-2 overflow-hidden rounded-full bg-muted">
                        <div
                          className={cn(
                            "h-full rounded-full transition-all",
                            riskBgColor(entity.risk_score)
                          )}
                          style={{
                            width: `${Math.min(entity.risk_score, 100)}%`,
                          }}
                        />
                      </div>
                    </div>
                    <p className="mt-2 text-xs text-muted-foreground">
                      {entity.alert_count} alert
                      {entity.alert_count !== 1 ? "s" : ""}
                    </p>
                  </Link>
                ))}
              </div>

              {/* Pagination */}
              <div className="mt-6 flex items-center justify-between">
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
