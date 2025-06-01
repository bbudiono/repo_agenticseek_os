# .cursorrules v2.0.0 - iOS/Swift Development Protocol
# Last Updated: 2025-05-31

---
## **P0 STOP EVERYTHING RULE (CRITICAL)**
*BUILD A POLISHED, BEAUTIFUL AND COHESIVE APPLICATION THAT IS FIT FOR PURPOSE.*

**ONLY difference between environments:** Visible "Sandbox" watermark in Sandbox build.
**Workflow:** Sandbox (develop/test) → Production (after ALL tests pass)
**Violation = P0 STOP EVERYTHING** - log and remediate immediately.

## **P0 CRITICAL MANDATES**
1. **NO PRODUCTION CODE EDITING** unless Sandbox has passed ALL tests
2. **USER MUST EXPLICITLY ACKNOWLEDGE** production code changes  
3. **SCRIPTS ARE PROJECT AGNOSTIC** - operate only within root folder hierarchy
4. **REFACTORING = TASK** - follow Sandbox→Production cycle, user must say "I ACCEPT...👍"
5. **Credentials in `{root}/.env`** - escalate to user if missing, BLOCK until resolved
6. **Build & Stability** - ensure best practices, builds correctly, accessible, follow TDD
7. **Code Alignment** - keep Sandbox/Production aligned except watermark
8. **GitHub Workflow** - commit after green builds
9. **FULL AUTOMATION** - NO PAUSING unless user provides actionable request
10. **CODE DOCUMENTATION** - ALL files need comments, rankings, overall rating >90%
11. **PATH VERIFICATION** - confirm correct project/folder/file before any operation
12. **COMPACT AT 30%** - summarize but detail current tasks for pickup

---

## Core Principles
- **Authority:** This document + `docs/BLUEPRINT.MD` are supreme directives
- **P0 STOP EVERYTHING:** Critical violation = immediate halt, log, escalate via SMEAC
- **SMEAC Format:** Standard for escalations/checkpoints
- **MCP Usage:** Meta-Cognitive Primitive tools integral to protocols
- **Sandbox-First:** All features validated in Sandbox before Production
- **Documentation Driven:** All actions traceable through docs

---

## 1. Governing Principles

### Protocol Adherence
- **Full Compliance:** Strict adherence to `.cursorrules` + `docs/BLUEPRINT.MD`
- **Conflict Resolution:** `BLUEPRINT.MD` for project params, this doc for operations
- **No Bypassing:** Rules cannot be bypassed without explicit override + user approval
- **Self-Correction:** Detect violations → stop → log → attempt fix → escalate if failed

### P0 STOP EVERYTHING Triggers
- Build Failure | Sandbox-First violation | Project Root violation
- Style/UX/UI compliance violations | Missing sandbox comments/watermarks
- Unauthorized doc creation/editing | Missing `DEVELOPMENT_LOG.MD` review
- Mock data in releases | Missing code complexity ratings

### Project Integrity
- **Master Path:** Project root from `docs/BLUEPRINT.MD` is canonical
- **Confined Operations:** ALL operations within project root only
- **Config Location:** `.env` at project root

### Build Stability (FOUNDATIONAL RULE #1)
- **KEEP BUILD GREEN AT ALL TIMES**
- Production code editing ONLY after passing tests
- Build failures = P0 CRITICAL priority
- Build verification after EVERY significant change

### iOS/Swift Excellence
- **Architecture:** MVVM + SwiftUI primary pattern
- **Structure:** Features/, Core/, UI/, Resources/ hierarchy  
- **Language:** camelCase variables, PascalCase types, async/await
- **UI:** SwiftUI first, SF Symbols, dark mode, SafeArea
- **Performance:** Regular Instruments profiling, lazy loading
- **Security:** Encrypt sensitive data, Keychain for credentials
- **Testing:** XCTest unit tests, XCUITest UI tests
- **P0 Violation:** Non-compliance triggers P0 STOP EVERYTHING

---

## 2. Operational Priority Loop

**MANDATORY Sequence:**
1. **Review `DEVELOPMENT_LOG.MD`** - latest context/blockers
2. **Test Production Build** (if status unknown)
3. **Review `ExampleCode/`** - analyze/suggest improvements
4. **FIX PRODUCTION BUILD** (if failing - P0 priority)
5. **General Testing** - smoke tests, consistency checks
6. **Enhance Build Prevention** - improve testing suite
7. **Implement Features** - Level 5+ tasks, Sandbox development
8. **Process Feature Inbox** - review, triage, sync tasks
9. **Process AI Recommendations** - convert to tasks
10. **Create Test Data** - aligned with requirements
11. **Review Scripts** - maintain, delete obsolete
12. **Proactive Stability** - enhance based on trends
13. **Assess Refactoring** - analyze opportunities

---

## 3. Development Workflow

### Task-Driven Development
- **STRICTLY FORBIDDEN:** Edit code without corresponding task in `docs/TASKS.MD`
- **No Work Without Task:** Every change traceable to task
- **Violation = CRITICAL BREACH:** Halt, log, escalate
- **Task Sync:** Keep synchronized across `~/docs/TASKS.md`, `~/tasks/tasks.json`

### AUTO ITERATE Mode
1. **Acknowledge & Plan** - Execute Deliberate Action Mandate
2. **Select & Assess** - ONE Level 4+ sub-task, break down if needed
3. **Implement Loop (TDD):**
   - Test First → Implement → Self-Review → Update Docs → Verify Build → Run Tests → Iterate
4. **Completion Check** - verify acceptance criteria met
5. **Checkpoint** - commit after green build/tests
6. **Report** - generate SMEAC checkpoint
7. **Promote** - carefully integrate Sandbox→Production

### SMEAC Checkpoint Format
```markdown
**VALIDATION REQUEST / CHECKPOINT**
- **PROJECT:** {ProjectName}
- **TIMESTAMP:** [YYYY-MM-DD HH:MM:SS UTC]
- **TASK:** [Task ID/Name from TASKS.MD]
- **STATUS:** [✅ Done / 🚧 In Progress / ⛔ Blocked / ❓ User Input Required]
- **KEY ACTIONS:** [actions, tools, scripts, build status, decisions]
- **FILES MODIFIED:** [list with paths]
- **DOCUMENTATION UPDATES:** [list]
- **BLOCKER DETAILS:** [if blocked]
- **USER ACTION REQUIRED:** [if needed - PRIORITY/questions/recommendations]
- **NEXT PLANNED TASK:** [next task]
```

---

## 4. Repository Structure

### Mandatory Structure
```plaintext
{repo}/ # NO STRAY FILES IN ROOT
├── docs/ # ALL documentation
│   ├── BLUEPRINT.MD # MASTER PROJECT SPEC
│   ├── DEVELOPMENT_LOG.MD # Canonical log
│   ├── TASKS.MD # Workflow hub
│   ├── BUILD_FAILURES.MD # Critical log
│   └── [other docs]
├── scripts/ # Global automation
├── tasks/tasks.json # Taskmaster.ai JSON
├── temp/ # Temporary files (gitignored)
├── _macOS/ # Platform root
│   ├── {ProjectName}/ # PRODUCTION
│   ├── {ProjectName}-Sandbox/ # SANDBOX
│   └── {ProjectName}.xcworkspace # SHARED
├── .cursorrules # This protocol
├── .env # Environment variables
└── CLAUDE.md # Claude memory
```

### Environment Segregation
- **Separate Projects:** Production/Sandbox maintain separate `.xcodeproj` files
- **Sandbox Comments:** EVERY sandbox file MUST include: `// SANDBOX FILE: For testing/development. See .cursorrules.`
- **Sandbox Watermark:** ALL sandbox apps MUST display visible SANDBOX UI watermark
- **Enforcement:** Automated checks, violation = P0 STOP EVERYTHING

---

## 5. Documentation & Code Quality

### Documentation Standards
- **Centralized:** ALL docs in `docs/`
- **Format:** Markdown only
- **Updates:** Key docs updated real-time
- **Context Review:** MUST review `DEVELOPMENT_LOG.MD` before ANY task (P0 violation if skipped)

### Code Quality Requirements
**MANDATORY Comment Block for ALL Files:**
```
* Purpose: [file's purpose/role]
* Issues & Complexity Summary: [known/anticipated issues]
* Key Complexity Drivers:
  - Logic Scope (Est. LoC): [~200]
  - Core Algorithm Complexity: [Med]
  - Dependencies: [3 New, 1 Mod]
  - State Management Complexity: [Med]
  - Novelty/Uncertainty Factor: [Low]
* AI Pre-Task Self-Assessment: [70%]
* Problem Estimate: [75%]
* Initial Code Complexity Estimate: [75%]
* Final Code Complexity: [78%]
* Overall Result Score: [92%]
* Key Variances/Learnings: [insights]
* Last Updated: [YYYY-MM-DD]
```

### Quality Metrics
- **Pre-Coding Assessment:** Formal complexity/risk assessment using MCPs
- **Post-Build Metrics:** Parse all files, calculate averages, log in `DEVELOPMENT_LOG.MD`
- **Refactoring Target:** ALL files <90% score must be refactored

---

## 6. Testing & Build Management

### Testing Mandates
- **TDD:** Tests written BEFORE implementation
- **Types:** Unit, Integration, E2E, UI, Performance, Security, Accessibility
- **Non-Destructive:** All tests must be safe and idempotent
- **Accessibility:** ALL UI elements programmatically discoverable/automatable (P0)

### Build Failure Protocol
**P0 CRITICAL - Immediate Actions:**
1. Use MCP `memory` to recall similar failures
2. Use MCP `sequential-thinking` for structured analysis  
3. Use MCP `perplexity-ask` to research solutions
4. Use MCP `context7` for latest documentation
5. Document in `docs/BUILD_FAILURES.MD`
6. Store plan in MCP `memory`

**Recovery Steps:** Detect → Log → Consult Guides → Run Diagnostics → Post-Mortem → Apply Fixes → Try Alternatives → Rollback → Restore → Re-verify → Escalate

---

## 7. MCP Server Usage (CRITICAL)

**MANDATORY:** Use appropriate MCPs at every major step. Log MCP used, rationale, outcome.

### Core Development MCPs
- **`applescript_execute`:** macOS automation, UI automation
- **`github`:** Repository operations, Git workflow
- **`filesystem`:** Secure file operations
- **`XcodeBuildMCP`:** Xcode project/build automation

### AI Enhancement MCPs  
- **`taskmaster-ai`:** Task management, multi-agent coordination
- **`sequential-thinking`:** Structured analysis, planning
- **`memory`:** Knowledge persistence, context retention
- **`context7`:** Context management via Upstash Redis

### Integration MCPs
- **`perplexity-ask`:** Web research, fact checking
- **`google-maps`:** Location services integration
- **`notionApi`:** Documentation sync

---

## 8. Documentation Reference

### Tier 1: CRITICAL (Required)
- **README.md:** Project overview, installation, quick start
- **CLAUDE.md:** AI agent guide, build commands, rules
- **TASKS.md:** Workflow hub, priorities, roadmap
- **ARCHITECTURE.md:** System design, tech stack

### Tier 2: BUILD & DEVELOPMENT
- **BUILD_FAILURES.md:** Build troubleshooting, failure analysis
- **DEVELOPMENT_LOG.md:** Canonical log of ALL actions/decisions
- **BUGS.md:** Known issues, workarounds, debugging

### Tier 3: CONFIGURATION
- **`.cursor/mcp.json`:** MCP server configuration
- **AI_MODEL_STM.MD:** AI reasoning, decisions, assessments
- **TECH_DEBT.md:** Technical debt tracking, refactoring plans

---

## Compliance Checklist (P0 COMPULSORY)

### Builds & Stability
- [ ] Production test build before new tasks
- [ ] All TDD work in Sandbox
- [ ] Sandbox testing complete before Production migration
- [ ] Build GREEN, all tests PASS

### Code Quality  
- [ ] Complete code blocks, no placeholders
- [ ] Modular codebase, consistent naming
- [ ] Code comments explaining "why" and "how"
- [ ] All files >90% rating with complexity comments

### Project Management
- [ ] 1 Prod `.xcodeproj`, 1 Sandbox `.xcodeproj`, 1 `.xcworkspace` only
- [ ] Correct file locations: Production in `{root}/_macOS/{ProjectName}/`
- [ ] Sandbox in `{root}/_macOS/{ProjectName-Sandbox}/`
- [ ] No redundant files in root

### Security & Data
- [ ] No mock data in Production
- [ ] Credentials in `{root}/.env`
- [ ] No hardcoded secrets
- [ ] Security audit after production features

### Process
- [ ] Review `DEVELOPMENT_LOG.MD` before tasks
- [ ] Task exists in `docs/TASKS.MD`
- [ ] Use TDD and proper dev cycle
- [ ] UI elements accessible/automatable
- [ ] Sandbox watermarking applied
- [ ] SMEAC checkpoint reports generated

### Xcode Settings
- [ ] Auto-manage signing enabled
- [ ] Bundle ID: `com.ablankcanvas.{ProjectName}`
- [ ] "Sign in with Apple" capability
- [ ] Minimum macOS: 13.5
- [ ] App Category: "Productivity"

---

## Glossary
- **SMEAC:** Situation, Mission, Execution, Administration, Command/Control
- **MCP:** Meta-Cognitive Primitive (AI agent tool)  
- **P0:** Priority 0 (Critical, triggers "P0 STOP EVERYTHING")
- **TDD:** Test-Driven Development
- **Canonical:** Single, authoritative source

## Quick Start
1. Link all code changes to tracked tasks
2. Follow Corporate Style Guide strictly  
3. Log all actions in `DEVELOPMENT_LOG.MD`
4. Escalate via SMEAC when blocked
5. Use compliance checklist before merging
6. Familiarize with MCP usage