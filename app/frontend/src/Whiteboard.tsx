// The system design surface: labeled boxes and arrows.
//
// React Flow is the right substrate here because the board is the thing the
// coach reads. A freehand canvas would look more like a whiteboard but would
// give the model pixels; a node graph gives it components, connections, and the
// annotations the user typed, which is what makes real critique possible.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Background,
  BackgroundVariant,
  Controls,
  Handle,
  MiniMap,
  Position,
  ReactFlow,
  addEdge,
  useEdgesState,
  useNodesState,
  type Connection,
  type Edge,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import "./whiteboard.css";
import type { domain } from "../wailsjs/go/models";

export type NodeKind =
  | "client"
  | "service"
  | "database"
  | "cache"
  | "queue"
  | "storage"
  | "cdn"
  | "balancer"
  | "external"
  | "note";

interface KindSpec {
  kind: NodeKind;
  label: string;
  icon: string;
  color: string;
}

/** The palette. Ordered roughly by how often a design reaches for each. */
export const KINDS: KindSpec[] = [
  { kind: "client", label: "Client", icon: "▤", color: "#8b93a7" },
  { kind: "balancer", label: "Load balancer", icon: "⇄", color: "#c99bf5" },
  { kind: "service", label: "Service", icon: "◆", color: "#6ea8fe" },
  { kind: "database", label: "Database", icon: "▮", color: "#5fd39a" },
  { kind: "cache", label: "Cache", icon: "⚡", color: "#f5b556" },
  { kind: "queue", label: "Queue", icon: "≣", color: "#f5896d" },
  { kind: "storage", label: "Object store", icon: "▣", color: "#5fc7d3" },
  { kind: "cdn", label: "CDN / edge", icon: "◎", color: "#a0b3ff" },
  { kind: "external", label: "3rd party", icon: "◈", color: "#98a1b3" },
  { kind: "note", label: "Note", icon: "✎", color: "#667085" },
];

const KIND_BY_ID = new Map(KINDS.map((k) => [k.kind, k]));

type BoardNodeData = {
  kind: NodeKind;
  label: string;
  detail: string;
  onChange: (id: string, patch: Partial<{ label: string; detail: string }>) => void;
};

/** One box. Label and detail are edited in place: opening a side panel to
 *  annotate a component costs more attention than the annotation is worth. */
function ComponentNode({ id, data, selected }: NodeProps) {
  const d = data as unknown as BoardNodeData;
  const spec = KIND_BY_ID.get(d.kind) ?? KIND_BY_ID.get("service")!;
  const isNote = d.kind === "note";

  return (
    <div
      className={`wb-node ${selected ? "selected" : ""} ${isNote ? "note" : ""}`}
      style={{ borderColor: selected ? spec.color : undefined }}
    >
      {!isNote && <Handle type="target" position={Position.Left} className="wb-handle" />}
      <div className="wb-node-head" style={{ color: spec.color }}>
        <span className="wb-node-icon">{spec.icon}</span>
        <span className="wb-node-kind">{spec.label}</span>
      </div>
      <input
        className="wb-node-label nodrag"
        value={d.label}
        placeholder="Name it"
        onChange={(e) => d.onChange(id, { label: e.target.value })}
      />
      <textarea
        className="wb-node-detail nodrag"
        value={d.detail}
        placeholder={isNote ? "Write it out" : "Annotate: choices, numbers, tradeoffs"}
        rows={isNote ? 4 : 2}
        onChange={(e) => d.onChange(id, { detail: e.target.value })}
      />
      {!isNote && <Handle type="source" position={Position.Right} className="wb-handle" />}
    </div>
  );
}

const nodeTypes = { component: ComponentNode };

export interface WhiteboardHandle {
  nodes: domain.BoardNode[];
  edges: domain.BoardEdge[];
}

interface Props {
  /** Called whenever the board changes, so the parent can push snapshots. */
  onChange: (board: WhiteboardHandle) => void;
}

let seq = 0;
const nextId = () => `n${Date.now().toString(36)}${(seq++).toString(36)}`;

export function Whiteboard({ onChange }: Props) {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [editingEdge, setEditingEdge] = useState<string | null>(null);
  const [edgeLabel, setEdgeLabel] = useState("");

  const patchNode = useCallback(
    (id: string, patch: Partial<{ label: string; detail: string }>) => {
      setNodes((current) =>
        current.map((n) => (n.id === id ? { ...n, data: { ...n.data, ...patch } } : n)),
      );
    },
    [setNodes],
  );

  // New boxes land on a grid wide enough that they never cover each other. A
  // small diagonal stagger looks tidier but buries the box you just added under
  // the next one, which is exactly when you want to type into it.
  const COL_WIDTH = 260;
  const ROW_HEIGHT = 190;
  const PER_ROW = 4;

  const addNode = useCallback(
    (kind: NodeKind) => {
      const id = nextId();
      setNodes((current) => {
        const i = current.length;
        return [
          ...current,
          {
            id,
            type: "component",
            position: {
              x: 60 + (i % PER_ROW) * COL_WIDTH,
              y: 60 + Math.floor(i / PER_ROW) * ROW_HEIGHT,
            },
            data: { kind, label: "", detail: "", onChange: patchNode },
          },
        ];
      });
    },
    [patchNode, setNodes],
  );

  const onConnect = useCallback(
    (c: Connection) => {
      setEdges((current) =>
        addEdge({ ...c, id: nextId(), label: "", animated: false, type: "smoothstep" }, current),
      );
    },
    [setEdges],
  );

  const deleteSelection = useCallback(() => {
    setNodes((current) => current.filter((n) => !n.selected));
    setEdges((current) => current.filter((e) => !e.selected));
  }, [setNodes, setEdges]);

  // Report upward on every change. The parent debounces before sending.
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;
  useEffect(() => {
    onChangeRef.current({
      nodes: nodes.map((n) => ({
        id: n.id,
        kind: (n.data as unknown as BoardNodeData).kind,
        label: (n.data as unknown as BoardNodeData).label,
        detail: (n.data as unknown as BoardNodeData).detail,
        x: n.position.x,
        y: n.position.y,
      })) as domain.BoardNode[],
      edges: edges.map((e) => ({
        id: e.id,
        source: e.source,
        target: e.target,
        label: typeof e.label === "string" ? e.label : "",
      })) as domain.BoardEdge[],
    });
  }, [nodes, edges]);

  const commitEdgeLabel = useCallback(() => {
    if (!editingEdge) return;
    setEdges((current) =>
      current.map((e) => (e.id === editingEdge ? { ...e, label: edgeLabel } : e)),
    );
    setEditingEdge(null);
    setEdgeLabel("");
  }, [editingEdge, edgeLabel, setEdges]);

  const unlabeled = useMemo(() => edges.filter((e) => !e.label).length, [edges]);

  return (
    <div className="wb-root">
      <div className="wb-palette">
        {KINDS.map((k) => (
          <button
            key={k.kind}
            className="wb-palette-btn"
            style={{ color: k.color }}
            onClick={() => addNode(k.kind)}
            title={`Add ${k.label}`}
          >
            <span className="wb-palette-icon">{k.icon}</span>
            <span>{k.label}</span>
          </button>
        ))}
        <div className="wb-palette-spacer" />
        <button className="wb-palette-btn danger" onClick={deleteSelection} title="Delete selected">
          <span className="wb-palette-icon">✕</span>
          <span>Delete</span>
        </button>
      </div>

      <div className="wb-canvas">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          nodeTypes={nodeTypes}
          onEdgeDoubleClick={(_, edge) => {
            setEditingEdge(edge.id);
            setEdgeLabel(typeof edge.label === "string" ? edge.label : "");
          }}
          fitView
          // Without a zoom cap, a board with two boxes fills the screen with
          // them and the text renders comically large.
          fitViewOptions={{ maxZoom: 1, padding: 0.2 }}
          minZoom={0.2}
          maxZoom={1.75}
          proOptions={{ hideAttribution: true }}
          deleteKeyCode={["Backspace", "Delete"]}
        >
          <Background variant={BackgroundVariant.Dots} gap={18} size={1} color="#242a37" />
          <Controls showInteractive={false} />
          <MiniMap pannable zoomable className="wb-minimap" nodeColor="#2b3244" maskColor="#0b0d12cc" />
        </ReactFlow>

        {editingEdge && (
          <div className="wb-edge-editor">
            <span className="faint">Label this connection</span>
            <input
              autoFocus
              value={edgeLabel}
              placeholder="e.g. gRPC, 40k writes/s, async"
              onChange={(e) => setEdgeLabel(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") commitEdgeLabel();
                if (e.key === "Escape") setEditingEdge(null);
              }}
            />
            <button className="primary" onClick={commitEdgeLabel}>
              Set
            </button>
          </div>
        )}

        {nodes.length === 0 && (
          <div className="wb-empty">
            <p>Add a component from the left to start.</p>
            <p className="faint">
              Drag from a box's right edge to connect it. Double-click a connection to label it.
            </p>
          </div>
        )}

        {unlabeled > 0 && (
          <div className="wb-hint faint">
            {unlabeled} unlabeled connection{unlabeled === 1 ? "" : "s"} - an unlabeled arrow tells a
            reviewer nothing
          </div>
        )}
      </div>
    </div>
  );
}
