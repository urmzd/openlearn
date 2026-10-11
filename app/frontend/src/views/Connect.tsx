// Connect: how to bring an agent to the course.
//
// The app has no assistant of its own. It serves the open course over MCP, and
// the learner connects the agent they already use. This view is the path from
// "I have an agent" to "it can see my course", in order.

import type { ReactNode } from "react";
import { AgentConnection } from "@/components/agent-connection";
import { PageBody, PageHeader, PageTitle } from "@/components/page";

function Step({ n, title, children }: { n: number; title: string; children: ReactNode }) {
  return (
    <li className="flex gap-4">
      <span className="flex size-6 shrink-0 items-center justify-center rounded-full border text-xs font-medium">
        {n}
      </span>
      <div className="min-w-0 flex-1 pb-8">
        <h2 className="text-sm font-medium">{title}</h2>
        <div className="mt-2 text-sm text-muted-foreground">{children}</div>
      </div>
    </li>
  );
}

const PROMPTS = [
  "What am I reading right now? Explain it in plain words.",
  "Quiz me on this page, one question at a time.",
  "Where am I on my path, and what should I do next?",
  "Open my attempt at the next exercise and check it. Don't fix it for me.",
];

const CAN = [
  ["Read", "the course outline, any document, and search across all of them"],
  ["See", "the page open in this app, your path progress, and your session scores"],
  ["Check", "your exercise attempts with the course's own checker, and mark stages you finish"],
];

export function Connect() {
  return (
    <>
      <PageHeader crumbs={[{ label: "Connect an agent" }]} />
      <PageBody>
        <PageTitle
          title="Connect an agent"
          description="Use the assistant you already have. It reads this course and checks your work through the app."
        />
        <ol>
          <Step n={1} title="Add openlearn to your agent">
            <p className="mb-3">
              Any agent that speaks MCP works: Claude Code, Claude Desktop, Cursor, and others. Copy
              the one that matches yours.
            </p>
            <AgentConnection />
          </Step>
          <Step n={2} title="Restart the agent, or start a new conversation">
            <p>
              It picks up the connection when it starts. It follows whichever course is open here, and
              it works whether or not this window is open.
            </p>
          </Step>
          <Step n={3} title="Ask it something about your course">
            <ul className="flex flex-col gap-1.5">
              {PROMPTS.map((p) => (
                <li key={p} className="rounded-lg bg-muted px-3 py-2 text-foreground">
                  {p}
                </li>
              ))}
            </ul>
          </Step>
        </ol>

        <div className="rounded-xl border p-4">
          <div className="mb-3 text-xs font-medium text-muted-foreground">What a connected agent can do</div>
          <dl className="flex flex-col gap-2 text-sm">
            {CAN.map(([verb, what]) => (
              <div key={verb} className="flex gap-3">
                <dt className="w-12 shrink-0 font-medium">{verb}</dt>
                <dd className="text-muted-foreground">{what}</dd>
              </div>
            ))}
          </dl>
          <p className="mt-3 text-xs text-muted-foreground">
            It cannot decide that an attempt is right or force a stage that has a check: those
            verdicts come from the course.
          </p>
        </div>
      </PageBody>
    </>
  );
}
