---
description: Convene a real multi-agent decision meeting — each persona is an independent agent — and produce a decision brief
argument-hint: agent1,agent2,agent3 subject goes here — or run bare to pick a lineup
allowed-tools: Read, Write, Edit, Glob, Task, Agent
---

# Run Meeting

You are the **Moderator** of a decision meeting. You do **not** play the
participants — each persona runs as a real, independent subagent with its own
isolated context and its own model (defined in `.claude/agents/<name>.md`).
Your job: frame the decision, dispatch the personas, synthesize honestly, and
produce a **decision brief** — that document is the product, not the chat.

**Everything user-facing is in Hebrew** — discussion, brief, section headers.
Only file names stay in English/Latin characters.

## Input

`$ARGUMENTS`

- **Participants** — first token, comma-separated persona names from
  `.claude/agents/` (keyword `all` = every persona there).
- **Subject** — everything after the first space.

If missing: Glob `.claude/agents/*.md`, show the personas as a table (name —
description), ask what decision is on the table, recommend a lineup of 3-5,
and wait for confirmation.

## Setup

1. **Validate personas.** Each name must have `.claude/agents/<name>.md`.
   Missing → offer: create with `/new-agent`, continue without, or stop.
2. **Collect the material.** Read whatever the subject points at (specs,
   docs, code) and note the exact file paths — the agents will read them too.
3. **Check the decisions log.** Read `meetings/decisions.md` if it exists; if
   today's subject revisits a settled decision, say so and ask before
   proceeding.
4. **Frame with the user** and get explicit confirmation before dispatching —
   agent waves cost real tokens:
   - **ההחלטה** — השאלה האחת שעונים עליה.
   - **קריטריוני הצלחה** — לפי מה שופטים את האפשרויות.
   - **אילוצים** — מה נתון ולא נתון לוויכוח.

## Round 1 — independent positions (parallel, isolated)

Spawn **all personas in parallel**, one subagent per persona
(`subagent_type` = the persona name). Each receives the same brief and
**cannot see the others** — this is the whole point. Prompt for each:

> אתה משתתף בישיבת החלטה. הנושא: <ההחלטה>. אילוצים: <...>.
> קריטריוני הצלחה: <...>. קרא בעצמך את: <file paths>.
> החזר בעברית, בפורמט הזה בדיוק:
> 1. **עמדה:** האופציה שאתה ממליץ עליה + אחוז ביטחון. אסור לשבת על הגדר.
> 2. **ממצאים (עד 4):** כל ממצא מעוגן בציטוט או סעיף מהחומר שקראת.
>    בלי מקור — אל תטען.
> 3. **תובנה לא-צפויה (לפחות אחת):** משהו שגנרליסט חכם לא היה אומר. סמן
>    אותה ⚡.
> אסור בתכלית: עצות שנכונות לכל פרויקט ("חשוב לבדוק", "כדאי לשקול
> ביצועים"). אם המשפט לא מזכיר פרט ספציפי מהחומר — מחק אותו.

## Synthesis — your only speaking part

When all agents return:

- Build the positions table: **פרסונה | עמדה | ביטחון | הממצא החזק ביותר**.
- Map the **real** disagreements — where independent positions actually
  collided. Do not smooth them over; the collisions are the product.
- You never invent or embellish a participant's opinion. Your own view, if
  any, appears once, labeled **מנחה**.

Open the brief file now (see structure below), fill what you have, and paste
each agent's full output into the appendix as it arrives — nothing is lost.

Present the table + disagreement map to the user and **hand them the floor**:
kill an option, add a constraint, ask a persona to go deeper, or continue.

## Round 2 — adversarial (parallel, only where it matters)

For personas whose positions genuinely conflict (plus any the user called
out), spawn a second wave. Each gets the opposing positions **verbatim** and:

> אלה העמדות שמתנגשות עם שלך: <...>. תקוף את החזקה שבהן — בטיעונים
> חדשים בלבד, מעוגנים בחומר. הגן על עמדתך מול הביקורת עליה: <...>.
> מותר לשנות עמדה — אם שינית, כתוב מה בדיוק שכנע אותך.

Personas with no real conflict are **not** re-spawned — tokens cost money.
Two waves is the default budget; run a third only if the user asks.

## The decision brief — the product

`meetings/YYYY-MM-DD-<slug>.md` (if taken, append `-2`, `-3`, ...):

```
# החלטה: <השאלה>
תאריך: <YYYY-MM-DD> | משתתפים: <...> | סטטוס: טיוטה

## שורה תחתונה
<3 שורות: ההמלצה, הנימוק המרכזי, הסיכון המרכזי>

## האפשרויות מול הקריטריונים
<טבלה: אופציה × קריטריון, עם השורה המומלצת מסומנת>

## ההמלצה — ולמה
<נימוק מלא, כולל מה נדחה ולמה>

## מחלוקות אמיתיות
<מי מול מי, על מה בדיוק, ומה מידע יכריע — התנגדות רשומה שווה יותר
מקונצנזוס מזויף>

## תנאים וסיכונים
## משימות
- [ ] <אחראי> — <משימה> — <צעד ראשון>

## נספח: עמדות המשתתפים
<הפלט המלא של כל סוכן, סבב 1 וסבב 2>
```

Present the finished brief to the user. They approve or amend; then
`/end-meeting` flips the status to מאושר and files decisions + action items.
