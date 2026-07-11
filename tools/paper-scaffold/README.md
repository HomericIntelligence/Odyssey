# Paper Scaffolding Tools

CLI tools for generating complete directory structures and boilerplate files for new paper implementations.

## Available Tools

### Paper Scaffolder (`scaffold.py`)

Generate a complete paper implementation directory with all boilerplate files.

**Language**: Python (justified by template processing, file generation)

### Usage

```bash
python tools/paper-scaffold/scaffold.py \
    --paper lenet5 \
    --title "LeNet-5: Gradient-Based Learning Applied to Document Recognition" \
    --authors "LeCun et al." \
    --year 1998 \
    --url "http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf" \
    --description "Convolutional neural network for handwritten digit recognition" \
    --output papers/
```text

### Generated Structure

```text
papers/lenet5/
├── README.md              # Paper documentation
├── model.mojo             # Model implementation stub
├── train.mojo             # Training script stub
├── data.mojo              # Data loading placeholder
├── metrics.mojo           # Metrics placeholder
├── tests/
│   └── test_model.mojo   # Test file stub
└── notes/
    ├── architecture.md    # Architecture notes
    └── results.md         # Results tracking
```text

## Templates

The scaffolder uses simple string substitution templates:

- `README.md.tmpl` - Paper documentation
- `model.mojo.tmpl` - Model implementation stub
- `train.mojo.tmpl` - Training script
- `test_model.mojo.tmpl` - Test file

### Template Variables

- `{{PAPER_NAME}}` - Short name (e.g., "lenet5")
- `{{PAPER_TITLE}}` - Full title
- `{{MODEL_NAME}}` - PascalCase model name (e.g., "Lenet5")
- `{{AUTHORS}}` - Paper authors
- `{{YEAR}}` - Publication year
- `{{PAPER_URL}}` - Link to paper
- `{{DESCRIPTION}}` - Brief description

## Design Principles

- **Consistency**: All papers follow the same structure
- **Simplicity**: Templates are minimal but functional
- **Flexibility**: Easy to customize generated files
- **Complete**: Includes README, code, tests, and notes

## Language Justification

Per [ADR-001](../../docs/adr/ADR-001-language-selection-tooling.md):

- **Why Python**: String templating, file I/O, no performance requirements
- **Conversion blocker**: Mojo regex not production-ready
- **Review**: Quarterly per ADR-001 monitoring strategy

## Future Enhancements

- Additional templates (eval.mojo, visualize.mojo)
- Template customization via config files
- Interactive mode with prompts
- Integration with data download scripts

## References

- [Issue #67](https://github.com/HomericIntelligence/Odyssey/issues/67): Tools planning
- [Issue #69](https://github.com/HomericIntelligence/Odyssey/issues/69): Tools implementation
- [ADR-001](../../docs/adr/ADR-001-language-selection-tooling.md): Language strategy
