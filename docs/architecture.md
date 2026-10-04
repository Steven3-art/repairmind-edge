# RepairMind Edge Architecture

## 1. Purpose

RepairMind Edge is an evidence-driven inspection system for electronic and edge devices.

Its architecture is designed around a simple principle:

> Perception should produce evidence. Evidence should drive decisions. Decisions may request new observations from the physical world.

The current implementation combines OpenCV 5 perception, immutable structured evidence, deterministic inspection decisions, electrical measurements, and auditable reporting.

The system intentionally avoids converting incomplete observations directly into hardware diagnoses.

---

## 2. System overview

```text
┌─────────────────────────────────────────────┐
│               Physical device               │
│                                             │
│           PCB / electronic system           │
└─────────────────────┬───────────────────────┘
                      │ image
                      ▼
┌─────────────────────────────────────────────┐
│          OpenCV 5 Perception Layer          │
│                                             │
│  • image validation                         │
│  • PCB localization                         │
│  • segmentation / contour processing        │
│  • geometric normalization                  │
│  • perspective processing                   │
│  • ROI proposal                             │
│  • visual artifact generation               │
└─────────────────────┬───────────────────────┘
                      │
                      │ structured observations
                      ▼
┌─────────────────────────────────────────────┐
│             Evidence Layer                  │
│                                             │
│  IMAGE_QUALITY                              │
│  BOARD_LOCALIZED                            │
│  BOARD_LOCALIZATION_UNCERTAIN               │
│  PERSPECTIVE_RECTIFIED                      │
│  VISUAL_REGION                              │
│  ELECTRICAL_MEASUREMENT                     │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│          Inspection Agent Layer             │
│                                             │
│  InspectionState                            │
│       ↓                                     │
│  deterministic decision policy              │
│       ↓                                     │
│  InspectionDecision                         │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
          ┌───────────────────────┐
          │   Inspection action   │
          │                       │
          │ REQUEST_OBSERVATION   │
          │ REQUEST_MEASUREMENT   │
          │ REPORT_FINDING        │
          └───────────┬───────────┘
                      │
                      │ when additional
                      │ physical evidence
                      │ is required
                      ▼
┌─────────────────────────────────────────────┐
│        Human / Physical Inspection          │
│                                             │
│  • another observation                      │
│  • electrical measurement                   │
│  • reference specification                  │
└─────────────────────┬───────────────────────┘
                      │ new evidence
                      ▼
              InspectionState
                      │
                      ▼
              next agent iteration
                      │
                      ▼
┌─────────────────────────────────────────────┐
│            Reporting Layer                  │
│                                             │
│  • inspection summary                       │
│  • decisions                                │
│  • measurements                             │
│  • evidence references                      │
│  • current inspection status                │
└─────────────────────────────────────────────┘
```

---

## 3. Architectural principles

### 3.1 Evidence before inference

RepairMind Edge distinguishes between:

```text
Observation
    ↓
Measurement
    ↓
Inference
    ↓
Recommendation
```

These stages must not be silently collapsed.

For example:

```text
Observed voltage: 4.927 V
```

is evidence.

It does not independently establish:

```text
Voltage is normal
```

because that conclusion requires an expected reference value and an acceptable tolerance.

The current agent therefore requests the missing reference rather than inventing one.

### 3.2 Explicit inspection state

Agent decisions operate on an explicit `InspectionState`.

The state contains:

- validated evidence;
- the current inspection iteration.

This makes each decision dependent on the evidence actually available at that point in the investigation.

### 3.3 Deterministic behavior

The current inspection policy is deterministic.

Equivalent inspection states produce equivalent decisions.

This provides:

- reproducibility;
- straightforward testing;
- clear debugging;
- auditable demonstrations.

Future reasoning components may become more sophisticated, but evidence provenance and explicit actions should remain architectural invariants.

### 3.4 Physical-world interaction

RepairMind Edge does not assume that all required information already exists in software.

An inspection decision may request a physical action such as an electrical measurement.

The result is converted into new structured evidence before the next reasoning iteration.

This creates the loop:

```text
Perceive
   ↓
Reason
   ↓
Request evidence
   ↓
Interact with physical system
   ↓
Acquire evidence
   ↓
Reason again
```

---

## 4. Perception layer

Implementation:

```text
src/repairmind_edge/perception.py
```

OpenCV 5 is the core technology of this layer.

Its responsibility is to transform an input PCB image into normalized visual information and structured evidence.

### 4.1 Input validation

The perception pipeline first verifies that the input can be processed safely.

Image characteristics such as dimensions, channels, brightness, and sharpness are represented through structured image-quality data.

### 4.2 PCB localization

The system attempts to isolate the dominant PCB from the surrounding scene.

Real PCB photographs can contain:

- cables;
- connectors;
- shadows;
- metallic surfaces;
- irregular boundaries;
- dense internal edges;
- background objects.

The current implementation therefore uses a multi-stage localization pipeline rather than relying on a single rectangular contour.

The localization subsystem includes segmentation, contour processing, morphology, boundary refinement, and geometric reasoning.

The result includes:

- whether the board was detected;
- localization confidence;
- board area ratio.

### 4.3 Geometric normalization

Once the board boundary is available, the PCB is transformed into a normalized representation.

This reduces perspective and geometric variability before downstream visual analysis.

The normalized image becomes the reference coordinate system for inspection regions.

### 4.4 ROI proposal

RepairMind Edge performs deterministic multi-scale analysis over the normalized board.

Candidate regions are evaluated using measurable local visual characteristics including:

- edge density;
- local contrast.

Spatial filtering is then applied using overlap suppression and center-distance constraints.

The current implementation limits the proposal set to a maximum of 12 regions.

These regions represent an inspection attention budget.

They do not represent:

- a complete component inventory;
- component classifications;
- calibrated defect probabilities.

### 4.5 Visual artifacts

The perception stage can generate:

```text
annotated.jpg
normalized.jpg
debug_contours.jpg
roi_overlay.jpg
```

These artifacts provide human-readable visibility into the same perception process that generates structured evidence.

---

## 5. Evidence layer

Implementation:

```text
src/repairmind_edge/models.py
```

RepairMind Edge uses typed Pydantic models to represent inspection data.

The current evidence categories are:

```text
IMAGE_QUALITY
BOARD_LOCALIZED
BOARD_LOCALIZATION_UNCERTAIN
PERSPECTIVE_RECTIFIED
VISUAL_REGION
ELECTRICAL_MEASUREMENT
```

A `VisualEvidence` object contains information such as:

- evidence identifier;
- evidence type;
- confidence or proposal score;
- description;
- optional source ROI;
- structured measurements.

Evidence objects provide the interface between perception and reasoning.

### Evidence identity

Evidence IDs make decision provenance explicit.

For example, the first decision in the real demonstration references the 12 visual-region evidence items that caused the agent to request an electrical measurement.

The second decision references:

```text
measurement_001
```

because that is the new evidence relevant to the second reasoning step.

---

## 6. Inspection agent

Implementation:

```text
src/repairmind_edge/agent.py
```

The inspection agent consumes an `InspectionState` and returns an `InspectionDecision`.

The current action types are:

```text
REQUEST_OBSERVATION
REQUEST_MEASUREMENT
REPORT_FINDING
```

### Current decision policy

The deterministic policy currently follows these major rules:

```text
Board localization uncertain
        ↓
REQUEST_OBSERVATION
        ↓
request a clearer full-board image
```

```text
Board localized
but no visual regions
        ↓
REQUEST_OBSERVATION
        ↓
request a closer focused image
```

```text
Visual regions available
but no electrical evidence
        ↓
REQUEST_MEASUREMENT
        ↓
request PCB supply voltage
```

```text
Electrical evidence available
but no reference specification
        ↓
REQUEST_OBSERVATION
        ↓
request expected value / tolerance
```

The current implementation therefore demonstrates evidence-driven progression without pretending that the agent has reached a component-level fault diagnosis.

---

## 7. Electrical evidence

Electrical measurements are represented independently from visual evidence generation.

The current measurement model records:

```text
quantity
value
unit
measurement_point
```

Example:

```text
quantity:          voltage
value:             4.927
unit:              V
measurement_point: board_supply
```

When accepted by the inspection workflow, this measurement is converted into evidence of type:

```text
ELECTRICAL_MEASUREMENT
```

The value is preserved as an observation.

No expected reference is inferred automatically.

---

## 8. Reporting layer

Implementation:

```text
src/repairmind_edge/reporting.py
```

The reporting layer consolidates validated inspection artifacts without introducing new technical conclusions.

The current report contains:

- report version;
- visual inspection summary;
- first agent decision;
- optional electrical measurement;
- optional second decision;
- high-level inspection status.

Current report statuses are:

```text
MEASUREMENT_REQUIRED
ADDITIONAL_EVIDENCE_REQUIRED
FINDING_AVAILABLE
```

The report is deterministic for equivalent validated inputs.

---

## 9. CLI orchestration

Implementation:

```text
src/repairmind_edge/cli.py
```

The command-line interface exposes the major inspection stages.

### Inspect

```text
repairmind inspect
```

Runs OpenCV perception and persists visual and structured artifacts.

### Reason

```text
repairmind reason
```

Loads structured evidence and selects the next inspection action.

### Measure

```text
repairmind measure
```

Adds an electrical observation and executes the next reasoning iteration.

### Demo

```text
repairmind demo
```

Orchestrates the reproducible end-to-end scenario.

The demo is intentionally built from the same inspection and reasoning stages available independently through the CLI.

---

## 10. Validated real inspection sequence

The current real PCB demonstration produces the following state transition:

```text
scanner_pcb.jpg
      │
      ▼
OpenCV 5 perception
      │
      ├── board detected
      ├── confidence ≈ 0.863
      ├── 12 visual regions
      └── 15 perception evidence items
      │
      ▼
InspectionState(iteration=1)
      │
      ▼
REQUEST_MEASUREMENT
      │
      └── target ROI: roi_012
      │
      ▼
Physical observation
      │
      └── 4.927 V @ board_supply
      │
      ▼
ELECTRICAL_MEASUREMENT
      │
      ▼
InspectionState(iteration=2)
      │
      ▼
REQUEST_OBSERVATION
      │
      └── expected supply reference required
      │
      ▼
inspection_report.json
      │
      ▼
ADDITIONAL_EVIDENCE_REQUIRED
```

The target `roi_012` is selected deterministically by the current policy.

It should not be interpreted as a semantic identification of the electrically optimal measurement point.

---

## 11. Generated artifacts

The canonical demo produces:

```text
artifacts/demo/
├── result.json
├── annotated.jpg
├── normalized.jpg
├── debug_contours.jpg
├── roi_overlay.jpg
├── decision_iteration_1.json
├── measurement.json
├── decision_iteration_2.json
└── inspection_report.json
```

These artifacts make the inspection sequence reproducible and auditable.

Generated artifacts are excluded from version control.

---

## 12. Testing and quality gates

The current project uses:

- pytest;
- Ruff;
- mypy;
- Pydantic validation.

At the current architecture checkpoint:

```text
pytest: 52 passed
Ruff:   passed
mypy:   passed
```

The test suite covers the perception pipeline, structured evidence, agent behavior, CLI orchestration, electrical measurement integration, and reporting.

---

## 13. Current trust boundaries

RepairMind Edge currently enforces several important boundaries.

### Visual activity is not fault probability

ROI confidence values describe the deterministic proposal mechanism.

They are not calibrated probabilities that a component is defective.

### Measurements are observations

An electrical measurement does not become a pass/fail result without sufficient reference information.

### Missing information is explicit

When required evidence is absent, the preferred action is to request that evidence.

### The system is assistive

RepairMind Edge is intended to support technical investigation.

It does not replace electrical safety procedures, measurement discipline, datasheets, or engineering judgment.

---

## 14. AWS integration boundary

AWS integration is not implemented at this architecture checkpoint.

It is the next functional milestone.

The intended responsibility boundary is:

```text
┌──────────────────────┐
│      OpenCV 5        │
│                      │
│ Visual perception    │
│ Evidence extraction  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ RepairMind Core      │
│                      │
│ Evidence             │
│ Decisions            │
│ Reports              │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│        AWS           │
│                      │
│ Cloud inspection     │
│ lifecycle            │
│ persistence          │
│ traceability         │
│ deployment           │
└──────────────────────┘
```

The exact AWS services will be documented only after they are implemented and validated.

This prevents the architecture documentation from claiming cloud capabilities that the current software does not yet provide.

---

## 15. Planned evolution

The next architecture milestones are:

```text
Current
  │
  ├── OpenCV 5 perception             ✓
  ├── structured evidence             ✓
  ├── deterministic inspection agent  ✓
  ├── electrical evidence             ✓
  ├── multi-step inspection loop      ✓
  ├── auditable reporting             ✓
  │
  ▼
AWS integration
  │
  ▼
inspection persistence / traceability
  │
  ▼
deployment hardening
  │
  ▼
hackathon demonstration
```

Longer-term research directions include:

- semantic relationships between ROIs and physical test points;
- expected-value and tolerance models;
- additional measurement modalities;
- known-good board comparison;
- component and functional-block understanding;
- oscilloscope and logic-analyzer evidence;
- datasheet/reference evidence;
- richer agentic reasoning.

The architectural invariant should remain:

> **RepairMind Edge must clearly distinguish what it observed, what it measured, what it inferred, and what it still needs to know.**