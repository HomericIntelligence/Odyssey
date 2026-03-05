# Tool Use Optimization

Efficient tool use reduces latency and token consumption.

## Parallel Tool Calls

**DO**: Make independent tool calls in parallel:

```python
# GOOD - Parallel reads
read_file_1 = Read("/path/to/file1.mojo")
read_file_2 = Read("/path/to/file2.mojo")
read_file_3 = Read("/path/to/file3.mojo")
# All three reads happen concurrently

# BAD - Sequential reads
read_file_1 = Read("/path/to/file1.mojo")
# Wait for result...
read_file_2 = Read("/path/to/file2.mojo")
# Wait for result...
read_file_3 = Read("/path/to/file3.mojo")
```

**DO**: Group related grep searches:

```python
# GOOD - Parallel greps
grep_functions = Grep(pattern="fn .*", glob="*.mojo")
grep_structs = Grep(pattern="struct .*", glob="*.mojo")
grep_tests = Grep(pattern="test_.*", glob="test_*.mojo")
# All searches run in parallel

# BAD - Sequential greps with waiting
grep_functions = Grep(pattern="fn .*", glob="*.mojo")
# Process results, then...
grep_structs = Grep(pattern="struct .*", glob="*.mojo")
```

## Bash Command Patterns

**DO**: Use absolute paths in bash commands (cwd resets between calls):

```bash
# GOOD - Absolute paths
cd /home/user/ProjectOdyssey && pixi run mojo test tests/shared/core/test_tensor.mojo

# BAD - Relative paths (cwd not guaranteed)
cd ProjectOdyssey && pixi run mojo test tests/shared/core/test_tensor.mojo
```

**DO**: Combine related commands with && for atomicity:

```bash
# GOOD - Atomic operation
cd /home/user/ProjectOdyssey && \
  git checkout -b 2549-claude-md && \
  git add CLAUDE.md && \
  git commit -m "docs: add Claude 4 optimization guidance"

# BAD - Multiple separate bash calls (cwd resets)
cd /home/user/ProjectOdyssey
git checkout -b 2549-claude-md  # Might run in different directory!
git add CLAUDE.md
```

**DO**: Capture output explicitly when needed:

```bash
# GOOD - Capture and parse output
cd /home/user/ProjectOdyssey && \
  pixi run mojo test tests/ 2>&1 | tee test_output.log && \
  grep -c PASSED test_output.log

# BAD - Output lost between calls
cd /home/user/ProjectOdyssey && pixi run mojo test tests/
# Output is gone, can't analyze it
```

## Tool Selection

Use the right tool for the job:

| Task | Tool | Rationale |
|------|------|-----------|
| Read file | Read | Fast, includes lines |
| Search pattern | Grep | Optimized regex |
| Find files | Glob | Fast discovery |
| Run commands | Bash | Execute shell |
| Edit lines | Edit | Precise replace |
| Write file | Write | Create/overwrite |

**DO**: Use the most specific tool:

```python
# GOOD - Use Glob to find files, then Read them
files = Glob(pattern="**/test_*.mojo")
for file in files:
    content = Read(file)

# BAD - Use Bash for file discovery
result = Bash("find . -name 'test_*.mojo'")
# Now have to parse shell output
```

## Agentic Loop Patterns

Claude Code supports iterative exploration through agentic loops. Use this pattern for complex tasks.

### Exploration -> Planning -> Execution

**Phase 1: Exploration** - Gather context and understand the problem:

- Read relevant documentation (CLAUDE.md, agent files, related issues)
- Search for existing patterns (grep for similar implementations)
- Identify constraints and requirements (compiler version, API patterns)
- Review recent changes (git log, PR history)
- Tools: Read, Grep, Glob, Bash (git log)

**Phase 2: Planning** - Design the solution:

- Break down the problem into subtasks
- Identify files to modify and create
- Design interfaces and data structures
- Plan verification steps (tests, linting, CI)
- Tools: Extended thinking, structured reasoning

**Phase 3: Execution** - Implement the solution:

- Make code changes (Edit, Write)
- Run verification (Bash: mojo test, pre-commit)
- Fix errors iteratively (Read error output → Edit → Rerun)
- Create PR and link to issue (gh-create-pr-linked skill)
- Tools: Edit, Write, Bash, agent skills

### Key Principles

1. **Iterate, don't perfect upfront** - Start with exploration, refine through execution
2. **Fail fast** - Run verification early and often
3. **Learn from errors** - Each failure provides information for the next iteration
4. **Checkpoint progress** - Commit working states, even if incomplete
5. **Adapt the plan** - If exploration reveals new constraints, update the plan

### When to Use Agentic Loops

- Complex refactoring across multiple files
- Debugging issues with unclear root causes
- Implementing features with design tradeoffs
- Exploring unfamiliar codebases

### When NOT to Use Agentic Loops

- Simple, well-defined tasks (use direct execution)
- Boilerplate code generation (use templates/examples)
- Mechanical changes (formatting, renaming)
