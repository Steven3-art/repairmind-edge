# RepairMind Edge - Technical Report

**Agentic Vision for Evidence-Driven Inspection and Troubleshooting of
Electronic and Edge Devices**

**Author:** Belinga Belinga Steve\
**Competition:** OpenCV AI Competition 2026, powered by AWS\
**Repository:** https://github.com/Steven3-art/repairmind-edge

------------------------------------------------------------------------

## 1. Executive Summary

RepairMind Edge is an experimental Agentic Vision system for
evidence-driven inspection and troubleshooting of electronic and edge
devices.

Electronic troubleshooting is rarely a single-image classification
problem. RepairMind Edge models troubleshooting as an explicit
multi-step loop:

``` text
Physical device
      |
      v
OpenCV 5 perception
      |
      v
Structured evidence
      |
      v
Inspection agent
      |
      v
Next observation or measurement
      |
      v
New evidence
      |
      v
Updated inspection decision
```

OpenCV 5 transforms PCB imagery into structured evidence. A
deterministic inspection agent consumes that evidence and selects an
explicit next action. Physical measurements can then become new evidence
and change the next decision.

The real-world demonstration uses a barcode-scanner PCB. The validated
workflow localizes the board, generates visual inspection regions,
requests an electrical measurement, incorporates an observed 4.927 V
value, and then requests the missing reference specification instead of
inventing a diagnosis.

The local workflow is complemented by AWS using authenticated API
Gateway access, Lambda validation, and private S3 artifact persistence.

RepairMind Edge is intentionally assistive rather than autonomous. It
distinguishes observations, measurements, inferences, and
recommendations.

## 2. Problem and Target Users

### 2.1 Problem

Electronic troubleshooting is iterative evidence gathering. A technician
may visually inspect a device, localize a board, identify areas
requiring attention, perform measurements, compare them with expected
behavior, and update the investigation.

RepairMind Edge treats computer vision as one evidence source inside
that larger loop rather than attempting to infer a defective component
directly from a single photograph.

For example, `Observed voltage: 4.927 V` is a measurement. Without an
expected value and tolerance, it does not establish that the voltage is
normal.

### 2.2 Target users

Target users include electronics technicians, repair engineers,
embedded-system developers, hardware prototyping teams, technical
laboratories, and field maintenance personnel.

The current implementation is a research and competition prototype, not
a production repair authority.

## 3. System Architecture

``` text
Physical Device
      |
      v
OpenCV 5 Perception
      |
      v
Structured Evidence
      |
      v
Inspection Agent
      |
      v
Inspection Action
      |
      v
Human / Physical Inspection
      |
      v
New Evidence -> Next Agent Iteration
      |
      v
Auditable Report
      |
      v
AWS Boundary
SigV4 -> API Gateway -> Lambda -> Private S3
```

OpenCV 5 performs visual perception. RepairMind Edge core models handle
evidence, decisions, and reporting. The human remains responsible for
requested physical measurements. AWS provides the cloud lifecycle around
validated inspection sessions and artifacts.

## 4. OpenCV 5 Perception Pipeline

OpenCV 5 is a core runtime dependency, not a presentation layer.

### 4.1 Image validation

The pipeline validates the input and derives characteristics including
dimensions, channels, brightness, and sharpness.

### 4.2 PCB localization

PCB photographs can contain cables, connectors, shadows, metallic
surfaces, irregular boundaries, dense internal edges, and background
objects. Localization therefore combines segmentation, contour
processing, morphology, boundary refinement, and geometric reasoning.

The result includes board detection, localization confidence, board area
ratio, and geometric boundary information when available.

### 4.3 Geometric normalization

The localized board is transformed into a normalized representation to
reduce geometric and perspective variability.

### 4.4 Deterministic visual-region proposal

Multi-scale analysis uses measurable characteristics including local
edge density, local contrast, spatial overlap, and center-distance
constraints. Spatial filtering suppresses redundant proposals.

The current implementation limits output to 12 regions. This is an
attention budget, not a component or defect count. ROI scores are
deterministic proposal scores, not calibrated fault probabilities.

### 4.5 Visual artifacts

``` text
annotated.jpg
normalized.jpg
debug_contours.jpg
roi_overlay.jpg
```

These artifacts expose perception results for inspection, debugging,
demonstration, and auditability.

## 5. Structured Evidence

Current evidence categories are:

``` text
IMAGE_QUALITY
BOARD_LOCALIZED
BOARD_LOCALIZATION_UNCERTAIN
PERSPECTIVE_RECTIFIED
VISUAL_REGION
ELECTRICAL_MEASUREMENT
```

Evidence objects preserve identifiers, type, confidence or proposal
score, description, source ROI where applicable, and structured
measurements. Agent decisions retain relevant evidence IDs for
traceability.

## 6. Agentic Vision Workflow

The inspection agent consumes an explicit `InspectionState` and returns
an `InspectionDecision`.

Supported actions are:

``` text
REQUEST_OBSERVATION
REQUEST_MEASUREMENT
REPORT_FINDING
```

### 6.1 Deterministic policy

Uncertain localization causes `REQUEST_OBSERVATION` for a complete sharp
image.

A localized PCB without visual regions causes `REQUEST_OBSERVATION` for
a closer focused image.

Visual regions without electrical evidence cause `REQUEST_MEASUREMENT`.

Electrical evidence without an expected reference or tolerance causes
`REQUEST_OBSERVATION` for the missing reference.

### 6.2 Why this is agentic

``` text
Visual regions
      |
      v
REQUEST_MEASUREMENT
      |
      v
New electrical evidence
      |
      v
Updated inspection state
      |
      v
REQUEST_OBSERVATION for reference
```

OpenCV-generated evidence therefore changes the subsequent action rather
than merely producing a fixed result for explanation.

## 7. Real-World Inspection Demonstration

The current demonstration uses a real barcode-scanner PCB.

### 7.1 Perception result

``` text
Board detected:          true
Localization confidence: approximately 0.863
Board area ratio:        approximately 0.747
Visual regions:          12
Perception evidence:     15 items
```

Localization confidence describes board localization, not defect
probability.

### 7.2 First agent iteration

``` text
Action:     REQUEST_MEASUREMENT
Target ROI: roi_012
```

`roi_012` is selected deterministically. It is not semantic
identification of an electrically optimal measurement point.

### 7.3 Physical measurement

``` text
Quantity:          voltage
Value:             4.927
Unit:              V
Measurement point: board_supply
```

The measurement point is supplied by the human operator. The observation
becomes `ELECTRICAL_MEASUREMENT` evidence.

### 7.4 Second agent iteration

The state now contains 16 evidence items. Because no expected supply
value or tolerance is available, the agent does not classify 4.927 V as
normal or abnormal.

It selects `REQUEST_OBSERVATION` for the expected supply voltage or
reference specification.

Final report status:

``` text
ADDITIONAL_EVIDENCE_REQUIRED
```

## 8. Auditable Reporting

The report consolidates the visual inspection summary, first decision,
optional measurement, optional second decision, and inspection status
without introducing additional conclusions.

Current statuses are:

``` text
MEASUREMENT_REQUIRED
ADDITIONAL_EVIDENCE_REQUIRED
FINDING_AVAILABLE
```

The real demonstration ends at `ADDITIONAL_EVIDENCE_REQUIRED`.

## 9. AWS Cloud Architecture

Computer vision and reasoning remain local. Validated reports and
selected artifacts can be published through AWS.

``` text
OpenCV 5 / Agent / Report
        |
        v
AWS Publisher
        |
        | SigV4 HTTPS
        v
API Gateway HTTP API
        |
        | AWS_IAM
        v
Lambda Python 3.12
        |
        +-- validate manifest/session/artifacts
        +-- derive S3 keys server-side
        +-- persist manifest
        +-- generate presigned PUT URLs
        |
        v
Private S3
```

### 9.1 Artifact transfer

Images do not pass through Lambda. Lambda validates the manifest and
returns presigned S3 PUT instructions. The client uploads approved
artifacts directly to private S3.

### 9.2 Published artifact allowlist

``` text
inspection_report.json
annotated.jpg
normalized.jpg
roi_overlay.jpg
```

### 9.3 Security controls

The cloud boundary uses SigV4, API Gateway `AWS_IAM`, Lambda validation,
server-derived S3 keys, private S3, Block Public Access, server-side
encryption, presigned uploads, and CloudWatch logging with explicit
retention. AWS credentials are not stored in the repository.

### 9.4 Current lifecycle limitation

The backend persists the session in an initial upload workflow. There is
no server-side finalize/verification endpoint yet. Successful client
PUTs must not be described as a server-verified finalized inspection
session.

## 10. Agentic Vision Evaluation

The deterministic evaluation suite tests orchestration behavior, not
component-level diagnostic accuracy.

### 10.1 Evaluation scenarios

  -----------------------------------------------------------------------
  ID                      Controlled state        Expected action
  ----------------------- ----------------------- -----------------------
  AV-01                   Uncertain PCB           `REQUEST_OBSERVATION`
                          localization

  AV-02                   Localized PCB without   `REQUEST_OBSERVATION`
                          inspection regions

  AV-03                   Visual regions without  `REQUEST_MEASUREMENT`
                          electrical evidence

  AV-04                   Electrical measurement  `REQUEST_OBSERVATION`
                          without reference
                          specification

  AV-05                   Non-actionable evidence `REQUEST_OBSERVATION`
  -----------------------------------------------------------------------

### 10.2 Results

``` text
Controlled scenarios:       5
Expected actions matched:   5
Failed scenarios:           0
Task success rate:          1.0
Unsupported diagnoses:      0
```

These results measure deterministic orchestration on five controlled
scenarios. They do not measure hardware fault-diagnosis accuracy.

### 10.3 Evidence-to-action trace

AV-03 verifies that visual evidence changes the action to
`REQUEST_MEASUREMENT`. AV-04 adds electrical evidence and verifies that
the next action changes to `REQUEST_OBSERVATION` for the missing
reference.

### 10.4 Safe fallback behavior

AV-05 verifies a safe `REQUEST_OBSERVATION` fallback rather than
`REPORT_FINDING`. The controlled evaluation records zero unsupported
diagnoses.

## 11. Testing and Engineering Quality

``` text
Core pytest suite:          77 passed
AWS backend pytest suite:   23 passed
Controlled total:          100 tests

Ruff:                       passed
mypy:                       passed
git diff --check:           passed
```

Tests cover perception, evidence, agent behavior, electrical measurement
integration, reporting, CLI orchestration, cloud session construction,
AWS publishing, Lambda API validation, and Agentic Vision evaluation.

The two pytest suites are reported separately before the controlled
total. This is not a claim of 100% code coverage or 100% diagnostic
accuracy.

## 12. Reproducibility

### 12.1 Validated environment

``` text
Python:   3.12.0
OpenCV:   5.0.0.93
NumPy:    2.5.3
Pydantic: 2.13.5
pytest:   9.1.1
boto3:    1.43.108
```

The project currently declares supported dependency ranges in
`pyproject.toml`. A final dependency-lock/pinning pass remains part of
submission hardening.

### 12.2 Installation

``` bash
git clone https://github.com/Steven3-art/repairmind-edge.git
cd repairmind-edge
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[aws,dev]"
```

### 12.3 Canonical local demonstration

Local sample images are intentionally excluded from version control.
With `samples/scanner_pcb.jpg` available:

``` powershell
repairmind demo samples\scanner_pcb.jpg `
    --output artifacts\demo `
    --measurement-value 4.927 `
    --measurement-unit V `
    --measurement-point board_supply
```

The exact public reproduction of the canonical image-based demonstration
still requires a judge-accessible demonstration asset or live
demonstration because the real sample image is currently local and
ignored.

### 12.4 Quality gates

``` bash
python -m pytest -q
python -m pytest infra/aws/tests/test_session_api.py -q
python -m ruff check src infra/aws/src infra/aws/tests
python -m mypy src
git diff --check
```

## 13. Failure Handling and Human Control

RepairMind Edge is designed around explicit uncertainty.

Localization uncertainty requests a better image. Missing visual detail
requests a closer image. Visual evidence without electrical evidence
requests a physical measurement. A measurement without a reference
requests the missing reference.

Physical measurements remain human-controlled. The system requests the
measurement, while a person determines how to interact safely with the
device and supplies the observed value.

## 14. Security and Observability

The session API uses AWS IAM authorization. The Lambda backend validates
manifests and controlled artifact types. S3 object keys are derived
server-side. Artifacts are stored in private S3 with public access
blocked, and controlled presigned PUT instructions are used for uploads.

CloudWatch provides Lambda logging with explicit retention. Evidence
IDs, decisions, measurements, JSON artifacts, and reports provide
application-level traceability.

## 15. Limitations

The current version does not:

-   identify a defective component from a single photograph;
-   classify visual ROIs as known electronic components;
-   treat ROI scores as calibrated fault probabilities;
-   infer an expected voltage that has not been supplied;
-   determine whether 4.927 V is normal without a reference and
    tolerance;
-   semantically identify the optimal electrical measurement point from
    `roi_012`;
-   replace datasheets, electrical safety procedures, or engineering
    judgment;
-   provide a server-side finalize/verification endpoint;
-   demonstrate broad diagnostic accuracy across multiple hardware
    families.

The real-world demonstration focuses on one PCB. The five-scenario
evaluation measures deterministic orchestration behavior, not hardware
fault-diagnosis accuracy.

## 16. Responsible Use

RepairMind Edge follows an evidence-first approach. It should
distinguish what was observed, what was measured, what was inferred,
what remains missing, and what should be inspected next.

When evidence is insufficient, requesting additional evidence is
preferable to manufacturing certainty.

Users remain responsible for safe handling of powered equipment, correct
use of measurement instruments, device-specific electrical limits,
authoritative documentation, and verification before repair actions.

RepairMind Edge is an inspection assistant, not an autonomous authority
for safety-critical electronic repair.

## 17. Real-World Impact

Hardware troubleshooting requires repeated movement between visual
inspection, measurements, documentation, and engineering judgment.
RepairMind Edge explores how those steps can become an auditable
evidence loop.

The architecture could extend to embedded controllers, IoT devices,
industrial edge equipment, laboratory prototypes, communications
hardware, and electronic repair workflows.

Future work includes semantic links between visual regions and test
points, known-good board comparison, expected-value/tolerance models,
functional-block understanding, oscilloscope evidence, logic-analyzer
evidence, and datasheet/reference evidence.

> RepairMind Edge must clearly distinguish what it observed, what it
> measured, what it inferred, and what it still needs to know.

## 18. Conclusion

RepairMind Edge demonstrates an evidence-driven approach to Agentic
Vision for electronic inspection.

OpenCV 5 does not simply produce an annotation for a chatbot to explain.
Its perception output becomes structured evidence that directly changes
the next inspection action.

``` text
PCB image
    |
    v
OpenCV 5 perception
    |
    v
15 perception evidence items
    |
    v
REQUEST_MEASUREMENT
    |
    v
4.927 V @ board_supply
    |
    v
16 evidence items
    |
    v
REQUEST_OBSERVATION
    |
    v
reference specification required
    |
    v
ADDITIONAL_EVIDENCE_REQUIRED
```

The controlled evaluation exercises five distinct evidence states, with
all five producing their expected actions and no unsupported diagnosis
in the evaluated scenarios.

AWS provides an authenticated persistence boundary using API Gateway,
Lambda, IAM authorization, and private S3 storage.

The current prototype deliberately stops where its evidence stops. The
objective is to select the next useful action while keeping
observations, measurements, decisions, and uncertainty auditable.
