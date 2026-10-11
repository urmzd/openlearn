// The command that connects an agent to this install, ready to copy.

import { useEffect, useState } from "react";
import { Check, Copy } from "lucide-react";
import { GetAgentCommand } from "../../wailsjs/go/main/App";
import { Button } from "@/components/ui/button";

/** Quotes one argument for a POSIX shell, only when it needs it. */
function quote(arg: string): string {
  if (/^[\w@%+=:,./-]+$/.test(arg)) return arg;
  return `'${arg.replace(/'/g, `'\\''`)}'`;
}

export function AgentConnection() {
  const [command, setCommand] = useState<string[]>([]);
  const [copied, setCopied] = useState("");

  useEffect(() => {
    GetAgentCommand()
      .then((c) => setCommand(c ?? []))
      .catch(() => setCommand([]));
  }, []);

  const snippets = [
    { label: "Claude Code", text: `claude mcp add openlearn -- ${command.map(quote).join(" ")}` },
    {
      label: "Any MCP client",
      text: JSON.stringify(
        { mcpServers: { openlearn: { command: command[0] ?? "", args: command.slice(1) } } },
        null,
        2,
      ),
    },
  ];

  const copy = (label: string, text: string) => {
    navigator.clipboard
      .writeText(text)
      .then(() => {
        setCopied(label);
        setTimeout(() => setCopied(""), 1500);
      })
      .catch(() => {});
  };

  return (
    <div className="flex flex-col gap-4">
      {snippets.map((s) => (
        <div key={s.label}>
          <div className="mb-1.5 flex items-center justify-between">
            <span className="text-xs font-medium text-muted-foreground">{s.label}</span>
            <Button variant="ghost" size="xs" onClick={() => copy(s.label, s.text)}>
              {copied === s.label ? <Check data-icon="inline-start" /> : <Copy data-icon="inline-start" />}
              {copied === s.label ? "Copied" : "Copy"}
            </Button>
          </div>
          <pre className="overflow-x-auto rounded-lg bg-muted p-3 font-mono text-xs leading-relaxed">
            {s.text}
          </pre>
        </div>
      ))}
    </div>
  );
}
