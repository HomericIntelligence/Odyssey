# References: gh-deduplicate-issues

## Raw Session Notes

### Context

- Repository: homericintelligence/ProjectOdyssey
- Date: 2026-03-08
- Starting open issue count: 683
- Ending open issue count: 570
- Issues closed: 114

### Issue Groups Processed

#### Group 1: ADR-009 Compliance/Enforcement

- Survivor: #3962 "Add test-count guard pre-commit hook per ADR-009 Phase 2"
- 79 duplicates closed: #4098, #4103, #4104, #4105, #4107, #4108, #4113, #4117, #4119,
  #4120, #4123, #4125, #4131, #4132, #4134, #4135, #4139, #4140, #4143, #4148, #4152,
  #4155, #4156, #4161, #4162, #4164, #4168, #4169, #4173, #4177, #4179, #4181, #4184,
  #4186, #4191, #4194, #4196, #4197, #4198, #4202, #4203, #4205, #4206, #4212, #4217,
  #4218, #4224, #4226, #4228, #4232, #4236, #4239, #4240, #4242, #4247, #4251, #4254,
  #4256, #4260, #4264, #4271, #4272, #4273, #4274, #4281, #4283, #4284, #4285, #4294,
  #4295, #4296, #4297, #4302, #4307, #4308, #4309, #4313, #4315, #4319, #4320

#### Group 2: ADR-009 Documentation

- Survivor: #3776
- 11 duplicates closed: #4007, #4121, #4170, #4178, #4204, #4207, #4225, #4243, #4255,
  #4275, #4282

#### Group 3: Remove continue-on-error Core Tensors

- Survivor: #4100
- 9 duplicates closed: #4147, #4160, #4163, #4190, #4199, #4211, #4227, #4259, #4318

#### Group 4: Core Utilities CI Splitting/Size

- Survivor: #4116
- 7 duplicates closed: #4114, #4144, #4195, #4250, #4265, #4266, #4268

#### Group 5: ADR-009 Apply Split to Other Files

- Survivor: #4150
- 4 duplicates closed: #4171, #4176, #4210, #4231

#### Group 6: Core Activations CI glob

- Survivor: #4157
- 1 duplicate closed: #4180

#### Group 7: Remove continue-on-error Core Loss

- Survivor: #4172
- 1 duplicate closed: #4304

#### Group 8: Negative Index __setitem__

- Survivor: #3387
- 2 duplicates closed: #3839, #4075

#### Group 9: validate_test_coverage track splits

- Survivor: #4109
- 1 duplicate closed: #4165

### Issues NOT Closed (Kept as Unique)

- #3358: detect 0-file patterns (unique feature)
- #4010: surface stale warnings in PR comment (unique feature)
- #4011: add --warn-stale / --error-stale flags (unique feature)
- #4012: handle multi-pattern strings (unique feature)
- #4298: use glob patterns (unique feature)

### Verification Results

All 9 survivors confirmed OPEN after execution.
Spot-checked closed issues #4098, #4320, #4282, #4318, #4075:

- All in CLOSED state
- All have "Duplicate of #XXXX" as last comment
- Close reason: "not planned"

### Execution Time

~8 minutes for 114 issues (228 API calls total)
Batched in groups of ~30 for progress visibility
