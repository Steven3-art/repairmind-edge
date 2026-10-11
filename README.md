# RepairMind Edge

**Agentic vision for evidence-driven inspection and troubleshooting of electronic and edge devices.**

RepairMind Edge is an experimental hardware inspection system that combines **OpenCV 5 perception**, structured evidence, deterministic agentic reasoning, and physical measurements to investigate electronic devices step by step.

Instead of attempting to diagnose a hardware fault from a single image, RepairMind Edge asks a more useful question:

> **Given the evidence available right now, what is the safest and most informative thing to inspect or measure next?**

The project is being developed for the **OpenCV AI Competition 2026, powered by AWS**.

> **Current status:** the OpenCV 5 inspection pipeline, evidence-driven troubleshooting loop, auditable reporting, and authenticated AWS inspection-session publishing workflow are implemented and validated.

---

## Why RepairMind Edge?

Electronic troubleshooting is inherently iterative.

A technician typically:

1. observes the device;
2. identifies relevant areas;
3. collects measurements;
4. compares observations with expected behavior;
5. updates the investigation;
6. decides what to inspect next.

RepairMind Edge models this workflow explicitly:

```text
Observation
    â†“
OpenCV 5 perception
    â†“
Structured evidence
    â†“
Inspection decision
    â†“
Physical observation / measurement
    â†“
New evidence
    â†“
Next inspection decision
```

The system deliberately separates **observation**, **measurement**, **inference**, and **recommendation**.

A visually interesting region is not automatically a defective component.

Likewise, a measured voltage is not automatically normal or abnormal unless an appropriate reference and tolerance are available.

---

## Current real-world demonstration

The current case study uses a real PCB from a barcode scanner under investigation.

RepairMind Edge processes the PCB photograph with OpenCV 5 and generates structured visual evidence.

On the validated demonstration image, the current pipeline produces:

- successful PCB localization;
- localization confidence of approximately `0.863`;
- geometric normalization of the board;
- `12` deterministic visual inspection regions;
- `15` structured perception evidence items.

The first agent iteration determines that visual evidence alone cannot establish electrical behavior:

```text
Action: REQUEST_MEASUREMENT
Target ROI: roi_012
```

The system requests a PCB supply-voltage measurement.

A real observation is then supplied:

```text
Quantity: voltage
Value:    4.927 V
Point:    board_supply
```

This creates new electrical evidence and triggers a second reasoning iteration.

Because no expected reference voltage or tolerance is available, RepairMind Edge does **not** classify `4.927 V` as good, bad, normal, or faulty.

Instead, it returns:

```text
Action: REQUEST_OBSERVATION

Next:
Provide the expected supply voltage or reference specification
for the measured point.
```

The resulting report status is:

```text
ADDITIONAL_EVIDENCE_REQUIRED
```

This behavior is intentional: insufficient evidence should produce a request for better evidence, not an invented diagnosis.

---

## Architecture

RepairMind Edge currently consists of four main layers:

```text
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚         Physical PCB / Image          â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                   â”‚
                   â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚       OpenCV 5 Perception Layer       â”‚
â”‚                                       â”‚
â”‚  â€¢ image validation                   â”‚
â”‚  â€¢ PCB localization                   â”‚
â”‚  â€¢ contour / segmentation processing  â”‚
â”‚  â€¢ geometric normalization            â”‚
â”‚  â€¢ perspective processing             â”‚
â”‚  â€¢ visual ROI proposal                â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                   â”‚
                   â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚       Structured Evidence Layer       â”‚
â”‚                                       â”‚
â”‚  â€¢ image quality                      â”‚
â”‚  â€¢ board localization                 â”‚
â”‚  â€¢ perspective evidence               â”‚
â”‚  â€¢ visual regions                     â”‚
â”‚  â€¢ electrical measurements            â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                   â”‚
                   â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚        Inspection Agent Layer         â”‚
â”‚                                       â”‚
â”‚  REQUEST_OBSERVATION                  â”‚
â”‚  REQUEST_MEASUREMENT                  â”‚
â”‚  REPORT_FINDING                       â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                   â”‚
                   â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚       Auditable Reporting Layer       â”‚
â”‚                                       â”‚
â”‚  â€¢ decisions                          â”‚
â”‚  â€¢ evidence IDs                       â”‚
â”‚  â€¢ measurements                       â”‚
â”‚  â€¢ inspection status                  â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

For a deeper technical description, see [`docs/architecture.md`](docs/architecture.md).

---

## Why OpenCV 5 is essential

OpenCV 5 is not a presentation layer or an optional dependency in RepairMind Edge.

It is the core perception engine responsible for transforming the physical PCB image into evidence that the inspection agent can reason about.

The current perception pipeline uses computer-vision processing for:

- image validation;
- PCB localization;
- segmentation and contour processing;
- geometric normalization;
- perspective rectification;
- edge and contrast analysis;
- deterministic ROI proposal;
- annotated inspection output;
- debugging visualizations.

Without this stage, the agent does not receive the board-localization and visual-region evidence required by the inspection workflow.

---

## Deterministic visual regions

After PCB normalization, RepairMind Edge performs multi-scale local analysis to identify visually active areas.

ROI proposals use measurable image characteristics such as:

- local edge density;
- local contrast;
- spatial overlap;
- center-distance suppression.

The system currently limits the inspection set to a bounded number of proposals.

These proposals are **not component classifications** and their scores are **not calibrated fault probabilities**.

They represent areas that may deserve further inspection.

---

## Evidence-driven agent

The inspection agent reasons over an explicit state containing validated evidence.

Its action model currently supports:

```text
REQUEST_OBSERVATION
REQUEST_MEASUREMENT
REPORT_FINDING
```

Each decision records the evidence IDs used during that iteration.

This creates an auditable relationship between perception and action.

The current demonstration implements a real two-iteration loop:

```text
PCB image
    â†“
15 perception evidence items
    â†“
Iteration 1
REQUEST_MEASUREMENT
    â†“
4.927 V @ board_supply
    â†“
16 available evidence items
    â†“
Iteration 2
REQUEST_OBSERVATION
    â†“
Expected reference specification required
```

---

## Quick start

### Requirements

- Python `3.12`
- OpenCV `5.0.0.93`
- Windows, Linux, or macOS with a supported Python/OpenCV environment

Clone the repository and enter the project directory:

```bash
git clone https://github.com/Steven3-art/repairmind-edge.git
cd repairmind-edge
```

Create a Python 3.12 virtual environment:

```bash
python -m venv .venv
```

Activate it.

PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install RepairMind Edge with development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[aws,dev]"
```

Verify the installation:

```bash
repairmind --help
```

---

## Run an inspection

Local sample images are intentionally excluded from Git.

Place your PCB photograph in `samples/`, for example:

```text
samples/scanner_pcb.jpg
```

Then run:

```bash
repairmind inspect samples/scanner_pcb.jpg --output artifacts/inspection
```

The inspection produces structured and visual artifacts including:

```text
artifacts/inspection/
â”œâ”€â”€ result.json
â”œâ”€â”€ annotated.jpg
â”œâ”€â”€ normalized.jpg
â”œâ”€â”€ debug_contours.jpg
â””â”€â”€ roi_overlay.jpg
```

---

## Run agent reasoning

After an inspection:

```bash
repairmind reason artifacts/inspection/result.json
```

The agent evaluates the available evidence and selects the next inspection action.

---

## Add an electrical measurement

Example:

```bash
repairmind measure artifacts/inspection/result.json \
    --quantity voltage \
    --value 4.927 \
    --unit V \
    --point board_supply
```

On PowerShell:

```powershell
repairmind measure artifacts\inspection\result.json `
    --quantity voltage `
    --value 4.927 `
    --unit V `
    --point board_supply
```

The observed value is preserved as evidence without being automatically interpreted as normal or abnormal.

---

## Run the end-to-end demo

The canonical RepairMind Edge demonstration is:

```powershell
repairmind demo samples\scanner_pcb.jpg `
    --output artifacts\demo `
    --measurement-value 4.927 `
    --measurement-unit V `
    --measurement-point board_supply
```

The complete demo generates:

```text
artifacts/demo/
â”œâ”€â”€ result.json
â”œâ”€â”€ annotated.jpg
â”œâ”€â”€ normalized.jpg
â”œâ”€â”€ debug_contours.jpg
â”œâ”€â”€ roi_overlay.jpg
â”œâ”€â”€ decision_iteration_1.json
â”œâ”€â”€ measurement.json
â”œâ”€â”€ decision_iteration_2.json
â””â”€â”€ inspection_report.json
```

The final `inspection_report.json` consolidates the inspection state without introducing conclusions that are not supported by the collected evidence.

---

## Engineering quality

RepairMind Edge uses automated quality gates throughout development.

Run the complete test suite:

```bash
pytest -q
```

Run linting:

```bash
ruff check .
```

Run static type checking:

```bash
mypy src
```

At the current milestone:

```text
Core tests:        77 passed
AWS backend tests: 23 passed
Controlled total:  100 tests
Ruff:              passed
mypy:              passed
```

These numbers describe the current development checkpoint and may increase as the project evolves.

---

## Project structure

```text
repairmind-edge/
â”œâ”€â”€ README.md
â”œâ”€â”€ pyproject.toml
â”œâ”€â”€ docs/
â”‚   â””â”€â”€ architecture.md
â”œâ”€â”€ samples/
â”‚   â””â”€â”€ README.md
â”œâ”€â”€ src/
â”‚   â””â”€â”€ repairmind_edge/
â”‚       â”œâ”€â”€ __init__.py
â”‚       â”œâ”€â”€ agent.py
â”‚       â”œâ”€â”€ cli.py
â”‚       â”œâ”€â”€ models.py
â”‚       â”œâ”€â”€ perception.py
â”‚       â””â”€â”€ reporting.py
â””â”€â”€ tests/
    â”œâ”€â”€ __init__.py
    â”œâ”€â”€ test_agent.py
    â”œâ”€â”€ test_cli.py
    â”œâ”€â”€ test_perception.py
    â””â”€â”€ test_reporting.py
```

Generated inspection artifacts and local experimental images are intentionally excluded from version control.

---

## Current limitations

RepairMind Edge is an experimental inspection system, not an autonomous electronic repair authority.

The current version does **not**:

- identify a defective component from a single image;
- treat visual ROI scores as fault probabilities;
- infer an expected voltage that has not been provided;
- interpret an electrical measurement without sufficient reference information;
- replace safe measurement procedures or engineering judgment.

The current ROI-to-measurement relationship is also deterministic rather than a semantic mapping to known electrical test points.

These limitations are explicit design boundaries for the current milestone.

---
## AWS integration

RepairMind Edge includes a real AWS-backed inspection-session publishing workflow.

The cloud boundary is deliberately separated from computer vision and inspection reasoning:

```text
OpenCV 5 perception
        |
        v
Structured visual evidence
        |
        v
Inspection agent
        |
        v
inspection_report.json
        |
        v
RepairMind AWS publisher
        |
        | AWS SigV4
        v
Amazon API Gateway HTTP API
        |
        | AWS_IAM authorization
        v
AWS Lambda
        |
        +-- validate the session manifest
        +-- derive S3 object keys server-side
        +-- persist the validated manifest
        +-- generate presigned upload instructions
        |
        v
Private Amazon S3
        |
        +-- inspection report
        +-- annotated image
        +-- normalized image
        +-- ROI overlay
## Roadmap

- [x] Real PCB perception pipeline
- [x] PCB localization and normalization
- [x] Deterministic visual ROI proposals
- [x] Structured visual evidence
- [x] Deterministic inspection agent
- [x] Electrical measurement evidence
- [x] Multi-iteration inspection loop
- [x] Auditable inspection report
- [x] Reproducible CLI demo
- [x] Authenticated AWS inspection-session publishing
- [ ] Public demonstration assets
- [ ] Hackathon deployment and submission hardening

---

## Responsible engineering

RepairMind Edge follows an evidence-first design principle.

The system should distinguish between:

- what was visually observed;
- what was physically measured;
- what can reasonably be inferred;
- what information is still missing;
- what should be inspected next.

When the available evidence is insufficient, requesting additional evidence is preferable to manufacturing certainty.

---

## Author

**Belinga Belinga Steve**

GitHub: `Steven3-art`

---

## License

This project is licensed under the MIT License.
