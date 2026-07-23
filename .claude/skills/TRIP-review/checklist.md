# Code Review Checklist

This file is the **single source of truth** for code-review criteria. Both human-driven reviews via `.claude/skills/TRIP-review` and Codex-driven reviews via `.claude/skills/codex-code-review` apply the criteria below — referenced, not copied — so the two review surfaces cannot drift.

## Systematic Review Checklist

### 1. Functional Requirements

- [ ] Implementation logic matches requirements correctly
- [ ] Interface/API matches documented specifications
- [ ] Error scenarios handled with proper feedback
- [ ] Edge cases and boundary conditions validated

### 2. Code Quality

- [ ] Proper typing (no unjustified dynamic types)
- [ ] DRY principle - no code duplication
- [ ] KISS principle - not unnecessarily complex
- [ ] Consistent, descriptive naming conventions
- [ ] Complex logic has explanatory comments
- [ ] Files/modules not excessively large
- [ ] Imports/includes organized, unused ones removed

### 3. Architectural Compliance

- [ ] Code follows established patterns from ARCHI.md
- [ ] Proper separation of concerns
- [ ] Appropriate abstractions used
- [ ] Consistent with existing codebase style

### 4. Detection & Calibration Correctness

- [ ] Color/contrast math is correct (relative contrast against substrate, not absolute RGB thresholds, unless deliberately deviating)
- [ ] Any new/changed calibration data columns are backward-compatible with existing `calibration_data.csv` rows, or a migration is documented
- [ ] Material-specific behavior (e.g. graphite's channel reliability findings) is not hardcoded into general detection logic — it should come from calibration data
- [ ] Thresholds/constants (area, tolerance, sampling radius/grid) are justified for the image scale they'll run at, not copy-pasted defaults
- [ ] Large-image memory handling respects or extends the strip-based processing pattern (no naive full-image float32 allocation for mosaic-scale inputs)

### 5. Image I/O & Viewer

- [ ] Analysis-path image loading avoids lossy formats (JPEG) for anything feeding color measurements
- [ ] 16-bit TIFF / OME-TIFF handling normalizes correctly rather than assuming 8-bit
- [ ] `ZoomViewer` changes remain usable via keyboard/trackpad only (no new mouse-only interaction)

### 6. Error Handling

- [ ] Errors are properly caught and handled
- [ ] Error messages are clear and actionable
- [ ] Failure modes are graceful
- [ ] Logging is appropriate (not too verbose, not silent)

### 7. Security (if applicable)

- [ ] Input validation implemented
- [ ] No sensitive data exposed
- [ ] Authentication/authorization respected
- [ ] No obvious vulnerabilities

### 8. Performance

- [ ] No obvious performance issues
- [ ] Resource cleanup implemented (no leaks)
- [ ] Appropriate data structures used
- [ ] No unnecessary operations in hot paths

---

## Issue Severity Classification

**Critical (Block Deployment)**:

- Security vulnerabilities
- Data corruption risks
- Breaking API/interface changes
- Authentication bypasses

**Major (Require Immediate Fix)**:

- Incorrect business logic
- Significant performance degradation
- Missing error handling
- Compilation/build errors

**Minor (Should Fix)**:

- Code style inconsistencies
- Missing documentation
- Code duplication
- Missing edge case handling

**Suggestions (Nice to Have)**:

- Performance optimizations
- Readability improvements
- Additional test coverage

---

## Review Completion Criteria (Approval Gate)

Minimum for approval:

- [ ] All functional requirements implemented
- [ ] No critical or major issues remaining
- [ ] `ruff check .` and `mypy .` pass clean
- [ ] Affected unit tests pass (`pytest <pattern>`, per the TRIP-2 testing gate)
- [ ] New logic has test coverage (or a coverage-debt ledger entry per the hard-to-cover policy)
- [ ] Documentation updated per project standards
