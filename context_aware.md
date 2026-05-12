I am building a project and want to maintain strict, reusable documentation that allows any new AI chat to fully understand the system without losing context.

Your task is to generate and maintain **three structured Markdown files**:

1. **project_context.md**
2. **changes.md**
3. **new_chat_prompt.md**

---

## 1. project_context.md (Single Source of Truth)

This file must contain a **complete, structured, and up-to-date system overview**.

Include the following sections:

### A. System Overview

* Project goal and scope
* Key architectural decisions
* Core design philosophy

### B. Tech Stack

* Backend: Python (FastAPI)
* ORM: SQLAlchemy
* Database: PostgreSQL
* Frontend: Streamlit
* Any additional tools/services

### C. Architecture

* High-level system design (modular breakdown)
* Component responsibilities
* Data flow between components

### D. API Design

For each major API:

* Endpoint name
* Purpose
* Request schema (high-level, no code)
* Response structure
* Dependencies
* Failure cases

### E. Core Modules & Functions

* Description of each module
* Responsibilities
* Key internal logic decisions (NO code)

### F. Data Model Decisions

* Key entities and relationships
* Important constraints
* Trade-offs made

### G. Critical Design Decisions

* Why certain approaches were chosen
* Alternatives that were rejected and why

### H. Known Limitations

* Current system weaknesses
* Scalability concerns
* Technical debt

---

## 2. changes.md (Strict Change Log)

This file must be updated **every time any change is made**.

Each entry MUST follow this format:

### Change #[Number]

* **What changed:**
* **Why it was changed:**
* **Previous approach:**
* **Problem in previous approach:**
* **New approach:**
* **Impact on system:**
* **Risks introduced:**
* **Lessons learned:**

Rules:

* Do NOT skip entries
* Do NOT summarize vaguely
* Be brutally honest about mistakes
* Capture reasoning, not just outcome

---

## 3. new_chat_prompt.md (Context Recovery System)

This file must be designed to allow a **completely new chat** to pick up the project with full clarity.

It must include:

### A. Instructions to the Model

* Read `project_context.md` first
* Then read `changes.md`
* Do NOT assume missing context
* Ask clarifying questions if anything is unclear

### B. Project Summary (Condensed)

* Short but precise overview of the system

### C. Critical Areas to Pay Attention To

* Important APIs
* Sensitive architectural decisions
* Known failure points

### D. Strict Rules

* Do NOT rewrite existing architecture without justification
* Do NOT ignore previous mistakes logged in changes.md
* Maintain consistency with existing design decisions

### E. Expected Behavior

* Think like a senior engineer
* Challenge decisions if needed
* Avoid repeating past errors

---

## Global Constraints

* Do NOT generate any code
* Do NOT be vague — everything must be explicit and structured
* Avoid unnecessary verbosity, but do NOT omit critical details
* Maintain consistency across all files
* Optimize for **long-term maintainability and context recovery**

---

## Workflow Rules

* Whenever I describe a new feature/change:

  1. Update `project_context.md` (only if architecture/design changes)
  2. ALWAYS append a new entry in `changes.md`
  3. Update `new_chat_prompt.md` only if context expectations change

---

Your goal is to make this documentation system so strong that:

* A completely new AI chat can continue the project without confusion
* No previous mistakes are repeated
* No critical context is lost across sessions
