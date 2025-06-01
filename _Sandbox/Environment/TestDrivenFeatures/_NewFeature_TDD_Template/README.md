# New Feature TDD Template

This template provides the complete structure for developing new features using the Sandbox TDD methodology. Each new feature should follow this exact pattern to ensure consistency and quality.

## Template Usage

1. **Copy this template** for each new feature:
   ```bash
   cp -r _Sandbox/Environment/TestDrivenFeatures/_NewFeature_TDD_Template/ \
         _Sandbox/Environment/TestDrivenFeatures/YourFeatureName_TDD/
   ```

2. **Or use the automated tool**:
   ```bash
   python _Sandbox/Tools/sandbox_tdd_runner.py --init-feature="YourFeatureName"
   ```

## TDD Phases

### 01_WriteTests/ (RED Phase)
- Write comprehensive failing tests before any implementation
- Include .cursorrules compliance tests
- Create user experience and accessibility tests
- Ensure all tests fail appropriately (RED state)

### 02_ImplementCode/ (GREEN Phase)  
- Implement minimal code to make all tests pass
- Focus on functionality first, then design system compliance
- Include accessibility from the start
- Achieve GREEN state (all tests passing)

### 03_RefactorImprove/ (REFACTOR Phase)
- Improve code quality while maintaining test passage
- Optimize design system compliance
- Enhance performance and user experience
- Polish accessibility features

### 04_ProductionReady/ (DEPLOY Phase)
- Prepare for production migration
- Complete all documentation
- Run final validation and quality gates
- Create migration plan

## .cursorrules Compliance Requirements

All features must comply with AgenticSeek design system:

### Color System
- Use `DesignSystem.Colors.*` exclusively
- No hardcoded color values
- Proper semantic color usage

### Typography
- Use `DesignSystem.Typography.*` for all text
- Maintain proper hierarchy
- Code vs chat text differentiation

### Spacing
- Follow 4pt grid system
- Use `DesignSystem.Spacing.*` values
- Semantic spacing for components

### Components
- Implement required ViewModifiers
- Agent interface compliance
- Accessibility integration

## Quality Gates

Before production migration, ensure:

- [ ] >95% test coverage achieved
- [ ] 100% .cursorrules compliance
- [ ] All accessibility requirements met
- [ ] Performance benchmarks achieved
- [ ] Documentation complete
- [ ] Integration tests pass
- [ ] Security review complete

## Success Metrics

Target metrics for TDD development:

- **TDD Cycle Time**: <4 hours for complete Red-Green-Refactor cycle
- **Design System Compliance**: 100% validation success
- **User Experience Score**: >4.8/5.0 across all personas
- **Accessibility Compliance**: 100% WCAG AAA compliance
- **Performance Impact**: <5% degradation in critical metrics

Follow this template structure for consistent, high-quality feature development in the AgenticSeek Sandbox environment.