// The sidebar. It answers three questions in order: where can I go, what am I
// working through, and what is there to read. Every list is flat, one line per
// item, the way a list of recent conversations is: a track opens in the main
// pane rather than unfolding into a tree here.
//
// Search lives at the top and replaces the lists with results while there is a
// query, so finding something never means leaving for another screen.

import { useEffect, useState } from "react";
import {
  BookOpen,
  ChevronsUpDown,
  Gauge,
  Home,
  Search as SearchIcon,
  Settings as SettingsIcon,
  Sparkles,
  SquareTerminal,
  Timer,
} from "lucide-react";
import { Search } from "../../wailsjs/go/main/App";
import type { course } from "../../wailsjs/go/models";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarInput,
  SidebarMenu,
  SidebarMenuBadge,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";
import type { Route } from "@/routes";

interface Props {
  info: course.Info | null;
  outline: course.Outline | null;
  paths: course.PathSummary[];
  route: Route;
  professorOpen: boolean;
  onNavigate: (route: Route) => void;
  onOpenDoc: (path: string) => void;
  onToggleProfessor: () => void;
}

export function AppSidebar({
  info,
  outline,
  paths,
  route,
  professorOpen,
  onNavigate,
  onOpenDoc,
  onToggleProfessor,
}: Props) {
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<course.Hit[] | null>(null);

  useEffect(() => {
    if (!query.trim()) {
      setHits(null);
      return;
    }
    let live = true;
    // The backend scans on every query; this only keeps it from being asked
    // on every keystroke.
    const t = setTimeout(() => {
      Search(query)
        .then((h) => live && setHits(h ?? []))
        .catch(() => live && setHits([]));
    }, 180);
    return () => {
      live = false;
      clearTimeout(t);
    };
  }, [query]);

  // Parts are reachable from the path that includes them; listing all of them
  // here buries the handful of paths a reader actually chooses between.
  const topPaths = paths.filter((p) => !p.part);
  const activeDoc = route.kind === "doc" ? route.path : "";

  const nav = [
    { label: "Home", icon: Home, route: { kind: "home" } as Route, show: true },
    { label: "Exercises", icon: SquareTerminal, route: { kind: "exercises" } as Route, show: !!info?.practice },
    { label: "Timed session", icon: Timer, route: { kind: "session", source: null } as Route, show: true },
    { label: "Progress", icon: Gauge, route: { kind: "progress" } as Route, show: true },
  ];

  return (
    <Sidebar>
      {/* The top padding clears the macOS window controls, which sit over the
          sidebar because the title bar is hidden. */}
      <SidebarHeader className="drag gap-2 pt-9">
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton
              size="lg"
              onClick={() => onNavigate({ kind: "settings" })}
              tooltip="Switch course pack"
            >
              <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-primary text-primary-foreground">
                <BookOpen className="size-4" />
              </span>
              <span className="grid min-w-0 flex-1 leading-tight">
                <span className="truncate font-medium">{info?.title ?? "OpenLearn"}</span>
                <span className="truncate text-xs text-muted-foreground">Course pack</span>
              </span>
              <ChevronsUpDown className="ml-auto size-4 text-muted-foreground" />
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
        <div className="relative">
          <SearchIcon className="pointer-events-none absolute top-1/2 left-2 size-3.5 -translate-y-1/2 text-muted-foreground" />
          <SidebarInput
            value={query}
            placeholder="Search the course"
            className="pl-7"
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Escape" && setQuery("")}
          />
        </div>
      </SidebarHeader>

      <SidebarContent>
        {hits ? (
          <SidebarGroup>
            <SidebarGroupLabel>
              {hits.length === 0 ? "No documents match" : `${hits.length} documents`}
            </SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                {hits.map((h) => (
                  <SidebarMenuItem key={h.path}>
                    <SidebarMenuButton
                      isActive={activeDoc === h.path}
                      onClick={() => onOpenDoc(h.path)}
                      className="h-auto flex-col items-start gap-0.5 py-1.5"
                      title={h.path}
                    >
                      <span className="w-full truncate">{h.title}</span>
                      <span className="line-clamp-2 w-full text-xs font-normal text-muted-foreground">
                        {h.snippet}
                      </span>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        ) : (
          <>
            <SidebarGroup>
              <SidebarGroupContent>
                <SidebarMenu>
                  {nav
                    .filter((n) => n.show)
                    .map((n) => (
                      <SidebarMenuItem key={n.label}>
                        <SidebarMenuButton
                          isActive={route.kind === n.route.kind}
                          onClick={() => onNavigate(n.route)}
                        >
                          <n.icon />
                          <span>{n.label}</span>
                        </SidebarMenuButton>
                      </SidebarMenuItem>
                    ))}
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>

            {topPaths.length > 0 && (
              <SidebarGroup>
                <SidebarGroupLabel>Paths</SidebarGroupLabel>
                <SidebarGroupContent>
                  <SidebarMenu>
                    {topPaths.map((p) => (
                      <SidebarMenuItem key={p.name}>
                        <SidebarMenuButton
                          isActive={route.kind === "path" && route.name === p.name}
                          onClick={() => onNavigate({ kind: "path", name: p.name })}
                          title={p.broken || p.title}
                          className="pr-14"
                        >
                          <span className="truncate">{p.title}</span>
                        </SidebarMenuButton>
                        <SidebarMenuBadge className="font-mono text-[11px] text-muted-foreground">
                          {p.broken ? "!" : `${p.done}/${p.total}`}
                        </SidebarMenuBadge>
                      </SidebarMenuItem>
                    ))}
                  </SidebarMenu>
                </SidebarGroupContent>
              </SidebarGroup>
            )}

            {[
              { label: "Tracks", tracks: outline?.tracks ?? [] },
              { label: "Reference", tracks: outline?.reference ?? [] },
            ]
              .filter((g) => g.tracks.length > 0)
              .map((g) => (
                <SidebarGroup key={g.label}>
                  <SidebarGroupLabel>{g.label}</SidebarGroupLabel>
                  <SidebarGroupContent>
                    <SidebarMenu>
                      {g.tracks.map((t) => (
                        <SidebarMenuItem key={t.dir}>
                          <SidebarMenuButton
                            isActive={
                              (route.kind === "track" && route.dir === t.dir) ||
                              activeDoc.startsWith(`${t.dir}/`)
                            }
                            onClick={() => onNavigate({ kind: "track", dir: t.dir })}
                          >
                            <span className="truncate">{t.title}</span>
                          </SidebarMenuButton>
                        </SidebarMenuItem>
                      ))}
                    </SidebarMenu>
                  </SidebarGroupContent>
                </SidebarGroup>
              ))}
          </>
        )}
      </SidebarContent>

      <SidebarFooter>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton isActive={professorOpen} onClick={onToggleProfessor}>
              <Sparkles />
              <span>Professor</span>
            </SidebarMenuButton>
          </SidebarMenuItem>
          <SidebarMenuItem>
            <SidebarMenuButton
              isActive={route.kind === "settings"}
              onClick={() => onNavigate({ kind: "settings" })}
            >
              <SettingsIcon />
              <span>Settings</span>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>
    </Sidebar>
  );
}
