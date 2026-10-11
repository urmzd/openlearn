# Postmortem rubric

`ol drill end` checks that a postmortem has the sections a drill asks for. This
rubric is the self-review that follows: answer each line y or n before you
share the document. It is blameless by construction: name systems and
decisions, never people.

- [ ] The summary says what users saw, for how long, and how many were affected.
- [ ] The timeline is in UTC and marks detection, mitigation, and resolution times.
- [ ] The root cause is a mechanism you can point to in code, configuration, or data.
- [ ] Detection says which alert fired (or why none did) and how long it took.
- [ ] Resolution says what was changed, and how you verified the fix held.
- [ ] Every action item has an owner, a ticket or file to change, and prevents a recurrence.
- [ ] Nothing in the document blames a person.
