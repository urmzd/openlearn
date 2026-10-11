// The sidebar. It answers three questions in order: where can I go, what am I
// working through, and what is there to read. Every list is flat, one line per
// item, the way a list of recent conversations is: a track opens in the main
// pane rather than unfolding into a tree here.
//
// Search lives at the top and replaces the lists with results while there is a
// query, so finding something never means leaving for another screen.
//
// Collapsing leaves a rail of icons rather than hiding the sidebar, so the one
// control that collapses it is also the one that brings it back. The pack
// switcher sits at the bottom, where a workspace switcher goes.

import { useEffect, useState } from "react";
import {
  BookOpen,
  Check,
  ChevronsUpDown,
  Gauge,
  Home,
  Search as SearchIcon,
  Settings as SettingsIcon,
  SquareTerminal,
  Timer,
} from "lucide-react";
import { Search } from "../../wailsjs/go/main/App";
import type { course, packs } from "../../wailsjs/go/models";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
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
  SidebarTrigger,
} from "@/components/ui/sidebar";
import type { Route } from "@/routes";

interface Props {
  info: course.Info | null;
  outline: course.Outline | null;
  paths: course.PathSummary[];
  route: Route;
  packList: packs.Pack[];
  onNavigate: (route: Route) => void;
  onOpenDoc: (path: string) => void;
  onSwitchPack: (ref: string) => void;
}

/** The open pack is matched by where it was read from: a directory for packs
 *  on disk, the id for built-in ones. */
export function isOpenPack(p: packs.Pack, info: course.Info | null): boolean {
  return (
    !!info &&
    p.id === info.id &&
    (info.builtIn ? p.source === "builtin" : p.ref === info.root)
  );
}

export function AppSidebar({
  info,
  outline,
  paths,
  route,
  packList,
  onNavigate,
  onOpenDoc,
  onSwitchPack,
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
    {
      label: "Exercises",
      icon: SquareTerminal,
      route: { kind: "exercises" } as Route,
      show: !!info?.practice,
    },
    {
      label: "Timed session",
      icon: Timer,
      route: { kind: "session", source: null } as Route,
      show: true,
    },
    {
      label: "Progress",
      icon: Gauge,
      route: { kind: "progress" } as Route,
      show: true,
    },
  ];

  return (
    <Sidebar collapsible="icon">
      {/* The top padding clears the macOS window controls, which sit over the
          sidebar because the title bar is hidden. */}
      <SidebarHeader className="drag gap-2 pt-9">
        <div className="flex h-8 items-center justify-between gap-2 pl-2 group-data-[collapsible=icon]:justify-center group-data-[collapsible=icon]:pl-0">
          <span className="truncate text-sm font-semibold tracking-tight group-data-[collapsible=icon]:hidden">
            {info?.title ?? "OpenLearn"}
          </span>
          <SidebarTrigger
            className="shrink-0 text-muted-foreground"
            title="Collapse or expand the sidebar"
          />
        </div>
        <div className="relative group-data-[collapsible=icon]:hidden">
          <SearchIcon className="pointer-events-none absolute top-1/2 left-2 size-3.5 -translate-y-1/2 text-muted-foreground" />
          <SidebarInput
            value={query}
            placeholder="Search"
            className="pl-7"
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Escape" && setQuery("")}
          />
        </div>
      </SidebarHeader>

      <SidebarContent>
        {hits ? (
          <SidebarGroup className="group-data-[collapsible=icon]:hidden">
            <SidebarGroupLabel>
              {hits.length === 0
                ? "No documents match"
                : `${hits.length} documents`}
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
                          tooltip={n.label}
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
              <SidebarGroup className="group-data-[collapsible=icon]:hidden">
                <SidebarGroupLabel>Paths</SidebarGroupLabel>
                <SidebarGroupContent>
                  <SidebarMenu>
                    {topPaths.map((p) => (
                      <SidebarMenuItem key={p.name}>
                        <SidebarMenuButton
                          isActive={
                            route.kind === "path" && route.name === p.name
                          }
                          onClick={() =>
                            onNavigate({ kind: "path", name: p.name })
                          }
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
                <SidebarGroup
                  key={g.label}
                  className="group-data-[collapsible=icon]:hidden"
                >
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
                            onClick={() =>
                              onNavigate({ kind: "track", dir: t.dir })
                            }
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
            <SidebarMenuButton
              isActive={route.kind === "settings"}
              onClick={() => onNavigate({ kind: "settings" })}
              tooltip="Settings"
            >
              <SettingsIcon />
              <span>Settings</span>
            </SidebarMenuButton>
          </SidebarMenuItem>
          <SidebarMenuItem>
            <DropdownMenu>
              <DropdownMenuTrigger
                render={<SidebarMenuButton tooltip="Switch course pack" />}
              >
                <BookOpen />
                <span className="truncate">
                  {info?.title ?? "No course pack"}
                </span>
                <ChevronsUpDown className="ml-auto text-muted-foreground" />
              </DropdownMenuTrigger>
              <DropdownMenuContent side="top" align="start" className="w-64">
                <DropdownMenuGroup>
                  <DropdownMenuLabel>Course packs</DropdownMenuLabel>
                  {packList.map((p) => (
                    <DropdownMenuItem
                      key={p.ref}
                      onClick={() => onSwitchPack(p.ref)}
                    >
                      <span className="truncate">{p.title}</span>
                      {isOpenPack(p, info) && <Check className="ml-auto" />}
                    </DropdownMenuItem>
                  ))}
                </DropdownMenuGroup>
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  onClick={() => onNavigate({ kind: "settings" })}
                >
                  Add or manage packs
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>
    </Sidebar>
  );
}
