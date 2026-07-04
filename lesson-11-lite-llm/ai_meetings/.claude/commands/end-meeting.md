---
description: End the current meeting — finalize the decision brief and file decisions and action items
allowed-tools: Read, Write, Edit, Glob
---

# End Meeting

Close out the decision brief in `meetings/` (the most recently updated one if
ambiguous — confirm with the user).

Everything written into the meeting files is in **Hebrew** — content and
section headers alike. Only file names stay in English.

1. **Finalize the brief.** Make sure it reflects the recommendation the user
   actually approved (including any amendments they made), the appendix holds
   all agent outputs, and every section is filled. Flip the header status:
   `סטטוס: טיוטה` → `סטטוס: מאושר`.

2. **Meeting or discussion?**
   - If there are **no action items**, this was a **discussion**: set the
     status to `סטטוס: דיון — ללא משימות` and skip step 3.
   - Otherwise it's a meeting — continue.

3. **File the action items.** Append them to `meetings/action-items.md`
   (create it if missing), tagged with the source meeting and date:

   ```
   ## מתוך: <נושא> (<YYYY-MM-DD>)
   - [ ] <אחראי> — <משימה> — <צעד ראשון>
   ```

4. **Update the decisions log.** Append every decision to
   `meetings/decisions.md` (create it if missing), one line per decision:

   ```
   - <YYYY-MM-DD> — <ההחלטה בשורה אחת> — ([<נושא>](<brief-file>))
   ```

   This file is the project's institutional memory: future meetings read it
   instead of re-reading every brief. Keep each line self-contained.

5. **Report back:** the brief path, whether it was logged as a meeting or a
   discussion, the count of action items filed, and the decisions added to
   the log.
