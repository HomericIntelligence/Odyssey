# Security Policy

## Reporting Security Vulnerabilities

**Do not open public issues for security vulnerabilities.**

We take security seriously. If you discover a security vulnerability, please report it responsibly.

## How to Report

### GitHub Security Advisory (Preferred)

Use GitHub's private vulnerability reporting:
**<https://github.com/HomericIntelligence/Odyssey/security/advisories/new>**

### What to Include

Please include as much of the following information as possible:

- **Description** - Clear description of the vulnerability
- **Impact** - Potential impact and severity assessment
- **Steps to reproduce** - Detailed steps to reproduce the issue
- **Affected versions** - Which versions are affected
- **Suggested fix** - If you have a suggested fix or mitigation

### Example Report

```text
Subject: [SECURITY] SQL Injection in data loader

Description:
The CSV data loader does not properly sanitize file paths, allowing
path traversal attacks.

Impact:
An attacker could read arbitrary files from the system.

Steps to Reproduce:
1. Create a CSV file with path "../../../etc/passwd"
2. Call load_csv("../../../etc/passwd")
3. Observe file contents are returned

Affected Versions:
v0.1.0 - v0.2.3

Suggested Fix:
Validate and canonicalize file paths before opening.
```

## Response Timeline

We aim to respond to security reports within the following timeframes:

| Stage | Timeframe |
| --- | --- |
| Initial acknowledgment | 48 hours |
| Preliminary assessment | 1 week |
| Fix development | Varies by severity |
| Public disclosure | After fix is released |

## Severity Assessment

We use the following severity levels:

| Severity | Description | Response |
| --- | --- | --- |
| **Critical** | Remote code execution, data breach | Immediate priority |
| **High** | Privilege escalation, data exposure | High priority |
| **Medium** | Limited impact vulnerabilities | Standard priority |
| **Low** | Minor issues, hardening | Scheduled fix |

## Responsible Disclosure

We follow responsible disclosure practices:

1. **Report privately** - Do not disclose publicly until a fix is available
2. **Allow reasonable time** - Give us time to investigate and develop a fix
3. **Coordinate disclosure** - We will work with you on disclosure timing
4. **Credit** - We will credit you in the security advisory (if desired)

## What We Will Do

When you report a vulnerability:

1. Acknowledge receipt within 48 hours
2. Investigate and validate the report
3. Develop and test a fix
4. Release a security update
5. Publish a security advisory
6. Credit the reporter (with permission)

## Scope

### In Scope

- ML Odyssey core library code
- Official examples and documentation
- Build and CI/CD infrastructure
- Dependencies (we will forward to upstream)

### Out of Scope

- Third-party integrations not maintained by us
- Social engineering attacks
- Physical security
- Issues already reported and being addressed

## Security Best Practices

When contributing to ML Odyssey:

- Never commit secrets, API keys, or credentials
- Use environment variables for sensitive configuration
- Validate and sanitize all external input
- Follow the principle of least privilege
- Keep dependencies updated

## Security Updates

Security updates are released as:

- Patch releases (e.g., v1.2.3 -> v1.2.4)
- Security advisories on GitHub
- Announcements on project communication channels

## Contact

For security-related questions that are not vulnerability reports:

- Open a GitHub Discussion with the "security" tag
- Use GitHub Security Advisories: <https://github.com/HomericIntelligence/Odyssey/security/advisories/new>

## Recognition

We appreciate the security research community's efforts to help keep ML Odyssey secure.
Reporters who follow responsible disclosure will be:

- Acknowledged in our security advisories
- Listed in our security hall of fame (coming soon)
- Thanked publicly (with permission)

## Privacy

For data-handling and GDPR guidance, see [docs/PRIVACY.md](docs/PRIVACY.md).

---

Thank you for helping keep ML Odyssey secure!
