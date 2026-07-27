# Security policy

## Supported versions

Fixes go into the latest release on PyPI. Older versions are not patched.

| Version | Supported |
| --- | --- |
| 0.2.x | yes |
| < 0.2 | no |

## Reporting a vulnerability

Report privately through [GitHub's private vulnerability
reporting](https://github.com/SaifPunjwani/polgrad/security/advisories/new), or by email
to saifpunjwani1230@gmail.com. Do not open a public issue for a vulnerability.

Include the polgrad version, the Python and PyTorch versions, and the smallest input
that reproduces the problem.

Expect an acknowledgement within seven days and an assessment within thirty. If a report
is confirmed, the fix ships in a patch release and the advisory is published with credit
unless you ask otherwise.

## Threat model

polgrad is a library of pure tensor functions. It has no network client, no server, no
deserialization of untrusted data, and no subprocess execution. The realistic security
surface is small, and reports in these areas are in scope:

- Anything in the installed package that reads the filesystem, the network, or the
  environment at import time. It should be nothing; a finding that it does is a bug.
- Supply chain problems with the published artifacts: a wheel or sdist on PyPI whose
  contents do not match the tagged source, or a dependency pin that resolves to
  something unexpected.
- The vendored upstream code under `src/polgrad/conformance/_vendor/` failing its
  SHA256 provenance check, which would mean the vendored bytes no longer match what the
  header claims.

Two things are out of scope. `tools/check_upstream_drift.py` fetches from GitHub over
HTTPS, but it is a development tool that is not part of the installed package and is
only ever run against the pinned upstream repositories. Numerical results that differ
from a framework are conformance findings, not vulnerabilities; open a normal issue for
those.
