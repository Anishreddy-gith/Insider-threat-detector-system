/**
 * Application shell – sidebar nav + top bar + outlet for pages.
 *
 * Professional SOC-style dark-theme layout with collapsible sidebar,
 * real-time connection indicator, and keyboard shortcuts.
 */

import { useState } from "react";
import { Outlet, NavLink, useNavigate } from "react-router-dom";
import {
  LayoutDashboard,
  Bell,
  Users,
  LogOut,
  Shield,
  Activity,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { useAuthStore } from "../store/authStore";
import { cn } from "../lib/utils";

const navItems = [
  { to: "/", icon: LayoutDashboard, label: "Dashboard" },
  { to: "/alerts", icon: Bell, label: "Alerts" },
  { to: "/entities", icon: Users, label: "Entities" },
  { to: "/model-health", icon: Activity, label: "Model Health" },
];

export default function Layout() {
  const user = useAuthStore((s) => s.user);
  const logout = useAuthStore((s) => s.logout);
  const navigate = useNavigate();
  const [collapsed, setCollapsed] = useState(false);

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      {/* ── Sidebar ─────────────────────────────────────────── */}
      <aside
        className={cn(
          "flex flex-col border-r bg-card transition-all duration-200",
          collapsed ? "w-16" : "w-64"
        )}
      >
        {/* Brand */}
        <div className="flex items-center gap-2 border-b px-4 py-4">
          <Shield className="h-6 w-6 shrink-0 text-primary" />
          {!collapsed && (
            <span className="text-lg font-bold tracking-tight">ITDS</span>
          )}
        </div>

        {/* Navigation */}
        <nav className="flex-1 space-y-1 px-2 py-4">
          {navItems.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              title={label}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
                  collapsed && "justify-center px-2"
                )
              }
            >
              <Icon className="h-4 w-4 shrink-0" />
              {!collapsed && label}
            </NavLink>
          ))}
        </nav>

        {/* User + collapse */}
        <div className="border-t px-2 py-3">
          {!collapsed && (
            <div className="mb-2 px-3 text-xs text-muted-foreground truncate">
              {user?.username ?? "unknown"} · {user?.role ?? ""}
            </div>
          )}
          <button
            onClick={handleLogout}
            title="Sign out"
            className={cn(
              "flex w-full items-center gap-3 rounded-md px-3 py-2 text-sm font-medium text-muted-foreground transition-colors hover:bg-destructive hover:text-destructive-foreground",
              collapsed && "justify-center px-2"
            )}
          >
            <LogOut className="h-4 w-4 shrink-0" />
            {!collapsed && "Sign out"}
          </button>

          <button
            onClick={() => setCollapsed((c) => !c)}
            className="mt-2 flex w-full items-center justify-center rounded-md py-1.5 text-muted-foreground hover:bg-accent hover:text-accent-foreground transition-colors"
            title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {collapsed ? (
              <ChevronRight className="h-4 w-4" />
            ) : (
              <ChevronLeft className="h-4 w-4" />
            )}
          </button>
        </div>
      </aside>

      {/* ── Main content ────────────────────────────────────── */}
      <main className="flex-1 overflow-y-auto p-6">
        <Outlet />
      </main>
    </div>
  );
}
