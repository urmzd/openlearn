// The sidebar. It answers three questions in order: where can I go, what am I
// working through, and what is there to read. Every list is flat, one line per
// item, the way a list of recent conversations is: a track opens in the main
// pane rather than unfolding into a tree here.
//
// Search lives at the top and replaces the lists with results while there is a
// query, so finding something never means leaving for another screen.
//
// Collapsing leaves a rail of icons rather than hiding the sidebar, so the one
// control that collapses it is also the one that brings it back. The brand
// leads, and the course switcher sits under it the way a workspace or project
// switcher does in the tools people already use.

import { useEffect, useState } from "react";
import {
  Check,
  ChevronsUpDown,
  Gauge,
  Home,
  PanelLeft,
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
  SidebarSeparator,
  SidebarTrigger,
  useSidebar,
} from "@/components/ui/sidebar";
import { Logo, LogoMark } from "@/components/logo";
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

/** The look of a section break in the sidebar: a small-caps label with a
 *  hairline running to the edge. Every group uses it, so the breaks between
 *  lists are quiet and identical. */
const BREAK =
  "h-7 gap-2 text-[11px] font-medium tracking-wider text-sidebar-foreground/50 uppercase after:h-px after:flex-1 after:bg-sidebar-border after:content-['']";

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
  const { toggleSidebar } = useSidebar();
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
        {/* Brand first, then the course under it: the app is OpenLearn, and
            the course is what you have open in it. In the rail the mark stands
            in for the row and turns into the expand control on hover. */}
        <div className="flex h-8 items-center justify-between pl-2 group-data-[collapsible=icon]:hidden">
          <Logo />
          <SidebarTrigger
            className="text-muted-foreground"
            title="Collapse sidebar"
          />
        </div>
        <button
          type="button"
          onClick={toggleSidebar}
          title="Expand sidebar"
          className="group/mark hidden size-8 items-center justify-center rounded-md text-sidebar-foreground outline-none group-data-[collapsible=icon]:flex hover:bg-sidebar-accent focus-visible:ring-2 focus-visible:ring-sidebar-ring"
        >
          <LogoMark className="group-hover/mark:hidden" />
          <PanelLeft className="hidden size-4 group-hover/mark:block" />
        </button>
        <DropdownMenu>
          <DropdownMenuTrigger
            title="Switch course"
            className="flex h-9 w-full items-center gap-2 rounded-lg border bg-background/40 px-2.5 text-left text-sm outline-none group-data-[collapsible=icon]:hidden hover:bg-sidebar-accent focus-visible:ring-2 focus-visible:ring-sidebar-ring"
          >
            <span className="flex size-5 shrink-0 items-center justify-center rounded bg-sidebar-accent text-[11px] font-semibold uppercase">
              {(info?.title ?? "?").charAt(0)}
            </span>
            <span className="min-w-0 flex-1 truncate font-medium">
              {info?.title ?? "No course"}
            </span>
            <ChevronsUpDown className="size-3.5 shrink-0 text-muted-foreground" />
          </DropdownMenuTrigger>
          <DropdownMenuContent
            side="bottom"
            align="start"
            className="w-(--anchor-width) min-w-56"
          >
            <DropdownMenuGroup>
              <DropdownMenuLabel>Courses</DropdownMenuLabel>
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
            <DropdownMenuItem onClick={() => onNavigate({ kind: "settings" })}>
              Add or manage courses
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
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
            <SidebarGroupLabel className={BREAK}>
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
                <SidebarGroupLabel className={BREAK}>Paths</SidebarGroupLabel>
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
                  <SidebarGroupLabel className={BREAK}>
                    {g.label}
                  </SidebarGroupLabel>
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

      <SidebarSeparator className="mx-0" />
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
        </SidebarMenu>
      </SidebarFooter>
    </Sidebar>
  );
}
