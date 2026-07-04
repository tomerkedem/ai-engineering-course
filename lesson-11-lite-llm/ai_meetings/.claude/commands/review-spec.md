---
description: Review a specification with independent expert agents and produce a verdict plus severity-ranked gap list
argument-hint: path/to/spec.md optional focus question
allowed-tools: Read, Write, Edit, Glob, Task, Agent
---

# Review Spec

Run a spec review using the `/run-meeting` engine — independent parallel
agents, grounded findings, a brief as the product — with the overrides below.
The output: a verdict (ready for development or not) and a severity-ranked
gap list.

## Input

`$ARGUMENTS`

- **Spec path** — the first token. If not found, Glob for close matches
  (`**/*<name>*`), show candidates, ask. No argument → ask for the path.
- **Focus question** — everything after the path (optional); if given, the
  review prioritizes it.

## Lineup

Default: `ux, pm, tester, critic`. Adjust to the spec's content:

- Touches data, auth, or permissions → add `security`.
- Touches infrastructure, integrations, or scale → add `architect` or `devops`.

Announce the lineup with a one-line reason; the user can override before the
agents are dispatched.

## Round 1 — independent findings hunt (parallel, isolated)

One subagent per persona, all in parallel, none sees the others. Prompt:

> אתה סוקר מסמך אפיון לפני פיתוח. קרא בעצמך את: <spec path> ואת כל קובץ
> שהוא מפנה אליו. <שאלת המיקוד אם יש>.
> החזר בעברית:
> 1. **פערים (עד 6):** לכל פער — ציטוט או סעיף מהאפיון (או ציון "חסר —
>    לא מופיע בכלל"), מה הבעיה, ודירוג חומרה: חוסם / גבוה / בינוני / נמוך.
>    - חוסם = אי אפשר להתחיל פיתוח; גבוה = יגרום לעבודה מחדש;
>      בינוני = כדאי לתקן; נמוך = שיפור רצוי.
> 2. **פסק דין:** מוכן / מוכן בתנאי / לא מוכן + נימוק במשפט.
> אסור: הערות שנכונות לכל אפיון. כל פער חייב להצביע על סעיף או היעדר
> ספציפי במסמך הזה.

## Synthesis

- **Merge and dedup** findings across agents; when agents disagree on
  severity, keep the higher one and note the dispute.
- Findings flagged by 2+ independent agents are marked **מאומת ×N** — they
  earned it, no second wave needed.

## Round 2 — contested findings only (parallel)

Only for findings where agents genuinely disagree (severity dispute, or one
agent's "blocker" that others ignored): send each disputed finding to the
persona best placed to judge it, with the conflicting views verbatim, and ask
for a ruling with new arguments. Skip this round entirely if nothing is
contested.

## The review brief — the product

`meetings/YYYY-MM-DD-review-<slug>.md`:

```
# סקירת אפיון: <שם המסמך>
תאריך: <YYYY-MM-DD> | סוקרים: <...> | סטטוס: טיוטה

## פסק דין
**מוכן / מוכן בתנאי / לא מוכן** — <נימוק בשורה>
<אם בתנאי — רשימת התנאים>

## טבלת הפערים
| # | חומרה | הפער | מקור באפיון | מי מצא |
<ממוין מחוסם לנמוך; "מאומת ×N" כשכמה סוכנים מצאו לבד>

## מחלוקות סוקרים
<פערים שנשארו במחלוקת ומה יכריע>

## נספח: ממצאי הסוקרים המלאים
```

Present the brief. Then follow `/end-meeting`: every **חוסם** and **גבוה**
finding becomes an action item; the verdict is recorded as the decision.
