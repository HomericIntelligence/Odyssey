# Notes Directory

## Overview

This directory contains supporting documentation for the ML Odyssey project.

**Important**: All implementation work and progress tracking is done through **GitHub issues**.
See `.claude/shared/github-issue-workflow.md` for the workflow.

## Repository Structure

```text
ProjectOdyssey/
├── notes/
│   └── blog/                    # Development blog entries
├── docs/
│   ├── adr/                     # Architecture Decision Records
│   └── dev/                     # Developer documentation
├── agents/                      # Agent system documentation
├── scripts/                     # Automation scripts
│   └── README.md                # Scripts documentation
└── logs/                        # Execution logs (not tracked)
```

## Documentation Locations

| Content | Location |
| --- | --- |
| Issue-specific work | GitHub issue comments |
| PR reviews | GitHub PR review comments |
| Architecture Decision Records | `/docs/adr/` |
| Developer documentation | `/docs/dev/` |
| Team guides | `/agents/` |
| Development blog | `/notes/blog/` |

## Working with GitHub Issues

### Reading Issue Context

```bash
# Get issue details and body
gh issue view <number>

# Get all comments on an issue
gh issue view <number> --comments

# Get specific fields
gh issue view <number> --json title,body,comments,labels
```

### Writing to Issues

```bash
# Short status updates
gh issue comment <number> --body "Status: [brief update]"

# Detailed implementation notes
gh issue comment <number> --body "$(cat <<'EOF'
## Implementation Notes

### Summary
[what was done]

### Files Changed
- path/to/file.mojo

### Verification
- [x] Tests pass
- [x] Linting passes
EOF
)"
```

## 5-Phase Development Process

Each component follows a 5-phase workflow:

1. **Plan** → Design and document the approach
2. **Test** → Write tests following TDD
3. **Implementation** → Build the functionality
4. **Packaging** → Integrate with existing code
5. **Cleanup** → Refactor and finalize

Each phase has its own GitHub issue with detailed instructions.

## Support

For questions or issues:

- Check the logs in `logs/`
- Review [scripts/README.md](../scripts/README.md)
- See existing issues on GitHub
- Refer to developer docs in `docs/dev/` and ADRs in `docs/adr/`
- Agent documentation in `agents/`
