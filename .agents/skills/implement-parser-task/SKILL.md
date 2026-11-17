---
name: implement-parser-task
description: Implement or improve task processing in AlexParser with regression tests and validation. Use when working on individual mathematical tasks or their processing pipeline.
---

# AlexParser Task Implementation

## Step 1. Understand

Read:

- AGENTS.md
- docs/ARCHITECTURE.md
- Current task specification
- Relevant existing implementation
- Relevant tests

Identify the actual variant and task number.

## Step 2. Plan

Before modifying code:

1. Define expected behavior.
2. Identify affected modules.
3. Identify regression risks.
4. Define acceptance criteria.
5. Prefer existing abstractions.

## Step 3. Implement

1. Write focused tests.
2. Reuse existing components.
3. Make minimal changes.
4. Preserve existing output contracts.
5. Avoid unrelated refactoring.

## Step 4. Verify

Run the relevant existing test commands.

Check:

- New tests
- Regression tests
- Data validation
- Mathematical formatting
- Output format
- No unintended changes to other tasks

Use mocks and existing fixtures instead of
making paid API requests.

## Step 5. Review

Inspect git diff.

Check:

- Unrelated changes
- Removed functionality
- Invalid assumptions
- Broken compatibility
- Secrets in modified files
- Unwanted API calls

## Step 6. Report

Provide:

1. Implemented functionality.
2. Modified files.
3. Tests executed and results.
4. Remaining risks.
5. Manual verification required.

## Restrictions

Never:

- Run paid AI APIs without approval.
- Delete existing caches.
- Modify .env.
- Rewrite unrelated modules.
- Perform commit or push without approval.


