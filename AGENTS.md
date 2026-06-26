# AGENTS.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```text
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## 5. Experiment Result Logging

**Every experiment must leave a dated Korean result note.**

When running or comparing experiments:

- Create a summary file under `experiments/`.
- Organize results by date and time, for example:
  `experiments/YYYY-MM-DD/HHMM_experiment_name.md`
- Write the experiment note in Korean.
- Record what changed, what stayed fixed, and what result came out.
- Keep the summary short enough to compare later.
- Include paths to related run outputs, configs, logs, and result CSVs.
- If an experiment was only planned but not run, mark it clearly as `미실행`.
- If results are missing or overwritten, say so instead of guessing.

Each experiment note should include:

```text
# [실험 이름]

- 날짜/시간:
- 목적:
- 변경한 것:
- 고정 조건:
- 데이터:
- 설정 파일:
- 결과 폴더:
- 주요 결과:
- 해석:
- 다음 액션:
```

Before reporting conclusions, check whether the result is based on:

1. actual files in `runs/` or `experiments/`,
2. Notion/GitHub notes only,
3. or inference from code/config changes.
