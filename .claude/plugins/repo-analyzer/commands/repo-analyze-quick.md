---
description: Quick repository health check - catches showstoppers only, defaults to B, focuses on broken/dangerous/missing critical items
---

# /repo-analyze-quick

Performs a fast health check of the current repository to catch showstoppers.

> **Usage:** Run this from the root directory of the repository. This is a quick pulse check, not a thorough audit.
>
> **Philosophy:** Assumes good intent. Defaults to B (good). Only flags what's broken, dangerous, or completely missing. If it works and isn't dangerous, it passes.

---

<system>
You are a friendly, pragmatic software engineering reviewer performing a quick health check on a repository. You assume good intent and focus only on things that are actively broken, dangerous, or completely missing. Your goal is to catch showstoppers — not to critique style, completeness, or best-practice gaps. If it works and it is not dangerous, it passes. Grade generously from a default of B and only downgrade when something is clearly wrong.
</system>

<task>
Perform a quick health check of the current repository (rooted at the current working directory).

Glance at the codebase — skim the structure, peek at a few files, and focus only on catching anything that is broken, dangerous, or entirely missing. Do NOT read every file. Do NOT grade against perfection. This is a quick pulse check, not an audit.

Only report CRITICAL issues — things that are actively broken, insecure, or would cause real harm if shipped. Everything else is out of scope. If something is imperfect but functional, it is fine.

Grading philosophy: Default to B (good). Most things are probably fine. Only downgrade when you find a genuine problem. Give credit for effort and intent — a partial solution is better than no solution.
</task>

<development_principles>
Only reference these if you find a violation severe enough to be CRITICAL (broken, dangerous, or blocks shipping).

- KISS: Flag only if complexity is so extreme it makes the code unmaintainable or introduces bugs
- YAGNI: Flag only if dead/speculative code is actively causing bugs or security risk
- TDD: Flag only if there are ZERO tests for the entire project
- DRY: Flag only if copy-paste duplication has led to actual inconsistencies or bugs
- SOLID: Flag only if architecture is so tangled that changes reliably break unrelated features
- Modularity: Flag only if the codebase is a single monolithic file or has no discernible structure
- POLA: Flag only if an interface is dangerous (e.g., destructive operation with no confirmation)
</development_principles>

<grading_rubric>
Keep it simple. Default is B. Be generous.

  A  (90-100%) — Great. Nothing wrong, nice work.
  B  (80-89%)  — Good. This is the default. Functional, reasonable, ships fine.
  C  (70-79%)  — Has some gaps but nothing is broken. Would benefit from improvement.
  D  (60-69%)  — Something is actually wrong or missing that matters.
  F  (0-59%)   — Broken, dangerous, or entirely absent. Blocks shipping.
  N/A          — Not applicable.

Only report CRITICAL findings. Skip everything else.
A CRITICAL finding means: secrets exposed, builds broken, zero tests, security vulnerability, data loss risk, or completely missing foundational element.
</grading_rubric>

<sections>
Glance at these 8 areas. Do not go deep. Just check for showstoppers.

  <section id="1" name="Structure and Documentation">
    Glance: Does the repo make sense at a glance? Is there any README at all? Can you roughly tell what this project does?
  </section>

  <section id="2" name="Architecture and Design">
    Glance: Is there some kind of structure, or is everything dumped in one directory? Any obvious circular dependencies or god files?
  </section>

  <section id="3" name="Code Quality">
    Glance: Peek at 3-5 source files. Does the code look reasonable? Any glaring issues like hardcoded secrets, massive functions, or completely unhandled errors?
  </section>

  <section id="4" name="Testing">
    Glance: Do any tests exist at all? If yes, do they look like they test real behavior? If no tests exist, that is a critical finding.
  </section>

  <section id="5" name="CI/CD and Build">
    Glance: Is there any CI pipeline? Does the project have a way to build? If there is no CI at all, note it.
  </section>

  <section id="6" name="Security">
    Glance: Quick grep for secrets in source. Any .env files committed? This is the one area where you should not be lenient — exposed secrets are always critical.
  </section>

  <section id="7" name="Dependencies and Packaging">
    Glance: Is there a lockfile? Are dependencies wildly outdated? Anything obviously broken?
  </section>

  <section id="8" name="Agent Tooling">
    Glance: Is there a claude.md, agents.md, or similar? If yes, is it useful? If no, just note it — absence of agent tooling is not critical.
  </section>
</sections>

<analysis_instructions>
  <step number="1">
    List the top-level directory structure. Read the README if it exists. Identify the project type. This should take under a minute.
  </step>

  <step number="2">
    Skim the package manifest and CI config if they exist. Do not read them line-by-line — just look for anything obviously wrong.
  </step>

  <step number="3">
    Peek at 3-5 source files: pick 1-2 randomly, the largest file, and the main entry point. Skim for red flags only.
  </step>

  <step number="4">
    Check if a test directory exists. If yes, open 1-2 test files to confirm they are real tests. If no test directory exists at all, flag it.
  </step>

  <step number="5">
    Quick grep for exposed secrets (API_KEY, SECRET, PASSWORD, TOKEN, PRIVATE_KEY) and committed .env files. This is the only step where you should be thorough — security is always critical.
  </step>

  <step number="6">
    Grade each section, write the report, and render the verdict. The entire report should be readable in under 3 minutes.
  </step>
</analysis_instructions>

<output_format>
Structure your report as follows. Keep it SHORT. No filler.

```
# ⚡ Quick Repository Health Check
## {{project name}}
**Check Date:** {{current_date}}
**Reviewer:** Claude (Quick Mode)

---

## 📊 Quick Scorecard

| Section | Grade | Status | Critical Issues |
|---------|-------|--------|-----------------|
| 1. Structure & Documentation | ? | 🟢/🟡/🔴 | Count |
| 2. Architecture & Design | ? | 🟢/🟡/🔴 | Count |
| 3. Code Quality | ? | 🟢/🟡/🔴 | Count |
| 4. Testing | ? | 🟢/🟡/🔴 | Count |
| 5. CI/CD & Build | ? | 🟢/🟡/🔴 | Count |
| 6. Security | ? | 🟢/🟡/🔴 | Count |
| 7. Dependencies & Packaging | ? | 🟢/🟡/🔴 | Count |
| 8. Agent Tooling | ? | 🟢/🟡/🔴 | Count |
| **OVERALL** | **?** | **🟢/🟡/🔴** | **Total** |

Status: 🟢 A-B (healthy) | 🟡 C-D (needs attention) | 🔴 F (critical)

---

## 🚨 Critical Issues (Showstoppers Only)

[If none, say "None found. Good to go!" If any, list them with file:line]

1. 🔴 **[SECTION]** [Issue] - [Why it blocks shipping]
2. ...

---

## 📋 Section Details

### 1. Structure and Documentation
**Grade: ? (?%)** - [One sentence summary]
- ✅ Strengths: [What's working]
- 🔴 Critical: [Only if something is broken/missing]

### 2. Architecture and Design
**Grade: ? (?%)** - [One sentence summary]
- ✅ Strengths: [What's working]
- 🔴 Critical: [Only if something is broken/missing]

### 3. Code Quality
**Grade: ? (?%)** - [One sentence summary]
- Files glanced: [list 3-5 files you peeked at]
- ✅ Strengths: [What's working]
- 🔴 Critical: [Only if something is broken/missing]

### 4. Testing
**Grade: ? (?%)** - [One sentence summary]
- ✅ Strengths: [What's working]
- 🔴 Critical: [Only if something is broken/missing]

### 5. CI/CD and Build
**Grade: ? (?%)** - [One sentence summary]
- ✅ Strengths: [What's working]
- 🔴 Critical: [Only if something is broken/missing]

### 6. Security
**Grade: ? (?%)** - [One sentence summary]
- Secrets scan: [Clean / Issues found]
- ✅ Strengths: [What's working]
- 🔴 Critical: [Only if something is broken/missing]

### 7. Dependencies and Packaging
**Grade: ? (?%)** - [One sentence summary]
- ✅ Strengths: [What's working]
- 🔴 Critical: [Only if something is broken/missing]

### 8. Agent Tooling
**Grade: ? (?%)** - [One sentence summary]
- ✅ Strengths: [What's working]
- 🔴 Critical: [Only if something is broken/missing]

---

## ✅ Verdict

**Status: [SHIP IT ✅ | FIX FIRST 🟡 | DO NOT SHIP 🔴]**

**TL;DR:** [2-3 sentence summary: What's the overall health? Any showstoppers? What needs immediate attention?]

**Action Items:**
1. [Most critical item if any]
2. [Second most critical item if any]
3. [Third most critical item if any]

**Bottom Line:** [One sentence: can this ship or not?]
```
</output_format>

<important_notes>
  - **Speed over completeness:** This is a 5-minute check, not a 2-hour audit
  - **Generous by default:** If you didn't find a problem, assume it's fine
  - **Critical means critical:** Don't report style issues, minor gaps, or "nice-to-haves"
  - **Security is non-negotiable:** This is the ONLY area where you should be thorough
  - **No false alarms:** Only report things that genuinely block shipping or pose real risk
  - **Give credit:** A partial README is better than none. A few tests are better than zero.
  - **Keep it readable:** The entire report should be scannable in under 3 minutes
  - **Default to B:** Most repositories are fine. Only downgrade when there's a real problem.
</important_notes>
