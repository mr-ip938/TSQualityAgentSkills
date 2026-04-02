---
name: ts-quality-agent
description: "Time series data quality assessment skill for pairwise comparison of two time series samples. Use this skill when users need to compare time series data quality, evaluate sample quality, or detect features such as missing values, noise, anomalies, trends, and periodicity. Core functionality: based on a multi-dimensional quality assessment framework (Bad Quality, Rare Pattern, Pattern Structure), performs pairwise comparison through ReAct loop tool calls, outputting winner determination and detailed explanation."
---

# TSQualityAgent - Time Series Quality Assessment Skill

## Overview

This skill provides procedural knowledge for time series pairwise comparison, helping agents systematically evaluate the quality of two time series samples. Through a structured quality dimension framework and tool invocation process, it ensures completeness and interpretability of the assessment.

## Core Task

**Input**: Two time series samples (series_A, series_B) and optional dataset description
**Output**: Pairwise comparison result `{ winner, confidence, explanation }`

## Assessment Workflow

```
Input Data (series_A, series_B, dataset_description)
        │
        ▼
┌─────────────────────────────────────┐
│  Phase 1: Perception & Planning      │
│  - Understand data characteristics   │
│  - Select quality dimensions to assess│
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  Phase 2: Inspection & Inference     │
│  - Execute ReAct loop for each dimension │
│  - Thought → Tool Call → Observation │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  Phase 3: Aggregation & Judgment     │
│  - Synthesize evidence across dimensions │
│  - Produce final judgment and explanation │
└─────────────────────────────────────┘
        │
        ▼
  Final Output: { winner, confidence, explanation }
```

---

## Phase 1: Perception & Planning

### 1.1 Data Perception

First, understand the basic characteristics of the input data:

1. **Data Length**: Whether the two series have consistent length
2. **Data Type**: Value range, presence of NaN values
3. **Data Description**: Context information from user-provided dataset_description

### 1.2 Dimension Selection

Select quality dimensions to assess based on data characteristics. **Not all dimensions need to be assessed** - avoid irrelevant inspections.

**Dimension Selection Decision Tree**:

| Condition | Select Dimension |
|-----------|------------------|
| NaN values present in series | `missing_value` |
| Obvious fluctuations in series | `noise_level` |
| User focuses on anomaly points | `rare_pattern` |
| Obvious trend in series | `trend` |
| Periodicity in series | `frequency` |
| Oscillatory characteristics | `amplitude` |
| Need to assess stability | `pattern_consistency` |

---

## Phase 2: Inspection & Inference

### 2.1 ReAct Loop

For each selected quality dimension, execute the following loop:

```
Thought: Analyze what needs to be detected for current dimension
   ↓
Action: Call corresponding tool
   ↓
Observation: Interpret tool return results
   ↓
[Loop until sufficient evidence for this dimension]
```

### 2.2 Quality Dimensions and Tool Mapping

#### Category 1: Bad Quality (Data Quality Issues)

| Dimension | Primary Tool | Auxiliary Tools | Judgment Criteria |
|-----------|--------------|-----------------|-------------------|
| `missing_value` | `missing_ratio` | — | Lower ratio is better |
| `noise_level` | `noise_profile` | `volatility`, `range_stats` | Lower noise_ratio is better |

**missing_value Assessment Process**:
1. Call `missing_ratio` for both A and B
2. Compare `missing_ratio` values
3. Lower value wins on this dimension

**noise_level Assessment Process**:
1. Call `noise_profile(window=5)` for both A and B
2. Compare `noise_ratio` (noise_std / signal_std)
3. Call `volatility` for auxiliary judgment if necessary
4. Lower noise_ratio wins on this dimension

#### Category 2: Rare Pattern (Anomaly Detection)

| Dimension | Primary Tools | Auxiliary Tools | Judgment Criteria |
|-----------|---------------|-----------------|-------------------|
| `rare_pattern` | `mad_residual_anomaly`, `contextual_anomaly` | `zscore_anomaly`, `outlier_density` | Moderate anomaly ratio is optimal |

**rare_pattern Assessment Process**:
1. First call `mad_residual_anomaly` for global anomaly detection
2. If context-dependent anomalies need detection, call `contextual_anomaly`
3. Compare `anomaly_ratio`:
   - Too many anomalies: data quality issue
   - Too few anomalies: may lack diversity
   - Moderate level is optimal

#### Category 3: Pattern Structure (Pattern Analysis)

| Dimension | Primary Tools | Auxiliary Tools | Judgment Criteria |
|-----------|---------------|-----------------|-------------------|
| `trend` | `trend_classifier` | `change_point_detector`, `stationarity_test` | Trend consistency |
| `frequency` | `seasonality_detector` | `autocorr` | Periodicity strength |
| `amplitude` | `cycle_amplitude` (periodic), `rolling_amplitude` (non-periodic) | `change_point_detector` | Amplitude consistency |
| `pattern_consistency` | `pattern_consistency_indicators` | `stationarity_test`, `change_point_detector` | Consistency indicators |

**trend Assessment Process**:
1. Call `trend_classifier` to get trend direction and strength
2. Compare `trend_strength`: moderate strength with clear direction is optimal
3. Call `stationarity_test` for auxiliary judgment if necessary

**frequency Assessment Process**:
1. Call `seasonality_detector` to detect periodicity
2. Compare `seasonal_strength`: stronger periodicity typically indicates better data quality
3. Use `autocorr` to verify periodic features

**amplitude Assessment Process**:
1. If series has periodicity, call `cycle_amplitude`
2. If no periodicity, call `rolling_amplitude`
3. Compare `amplitude_cv`: lower coefficient of variation indicates more consistent amplitude

**pattern_consistency Assessment Process**:
1. Call `pattern_consistency_indicators` to get consistency metrics
2. Focus on `lumpiness`, `flat_spots`, `crossing_points`
3. Call `change_point_detector` to detect structural changes if necessary

---

## Phase 3: Aggregation & Judgment

### 3.1 Evidence Aggregation

Summarize assessment results across dimensions into an evidence table:

| Dimension | Result A | Result B | Winner | Confidence |
|-----------|----------|----------|--------|------------|
| missing_value | ... | ... | A/B/tie | High/Medium/Low |
| noise_level | ... | ... | A/B/tie | High/Medium/Low |
| ... | ... | ... | ... | ... |

### 3.2 Final Judgment

Make comprehensive judgment based on evidence table:

1. **Count wins across dimensions**
2. **Consider dimension weights** (adjustable based on data type)
3. **Assess confidence level**

### 3.3 Output Format

```json
{
  "winner": "A" | "B" | "tie",
  "confidence": 0.0 ~ 1.0,
  "explanation": "Detailed reasoning including assessment results for each dimension and basis for final judgment"
}
```

---

## Tool Invocation Standards

### Parameter Conventions

- `series_name`: Must be "A" or "B"
- Call tools for A first, then B for easy comparison
- Maintain parameter consistency (e.g., window size)

### Return Value Interpretation

All tools return dict format with key fields:

| Tool | Key Field | Meaning |
|------|-----------|---------|
| `missing_ratio` | `missing_ratio` | Missing value ratio |
| `noise_profile` | `noise_ratio` | Noise ratio |
| `zscore_anomaly` | `anomaly_ratio` | Anomaly ratio |
| `trend_classifier` | `trend_strength` | Trend strength |
| `seasonality_detector` | `seasonal_strength` | Seasonality strength |
| `cycle_amplitude` | `amplitude_cv` | Amplitude coefficient of variation |

### Tool Selection Quick Reference

| Dimension | Primary Tools | Auxiliary Tools |
|-----------|---------------|-----------------|
| `missing_value` | `missing_ratio` | — |
| `noise_level` | `noise_profile`, `volatility` | `range_stats` |
| `rare_pattern` | `mad_residual_anomaly`, `contextual_anomaly` | `zscore_anomaly`, `outlier_density` |
| `trend` | `trend_classifier` | `change_point_detector`, `stationarity_test` |
| `frequency` | `seasonality_detector` | `autocorr` |
| `amplitude` | `cycle_amplitude` (periodic), `rolling_amplitude` (non-periodic) | `change_point_detector` |
| `pattern_consistency` | `pattern_consistency_indicators` | `stationarity_test`, `change_point_detector` |

---

## Tool Implementation Location

All tool function implementations for this skill are located in the `skills/ts-quality-agent/scripts/` directory. The structure is organized as follows:

- **`registry.py`**: Contains `TOOL_REGISTRY` mapping tool names to callable Python functions, and `TOOL_SCHEMAS` defining tool arguments for OpenAI function calling.
- **`bad_quality.py`**: Data integrity & noise tools
  - `missing_ratio`
  - `noise_profile`
  - `volatility`
  - `range_stats`
- **`rare_pattern.py`**: Anomaly detection tools
  - `zscore_anomaly`
  - `outlier_density`
  - `mad_residual_anomaly`
  - `contextual_anomaly`
- **`pattern_structure.py`**: Structural analysis tools (trend, periodicity, amplitude, consistency)
  - `trend_classifier`
  - `seasonality_detector`
  - `change_point_detector`
  - `pattern_consistency_indicators`
  - `stationarity_test`
  - `autocorr`
  - `rolling_amplitude`
  - `cycle_amplitude`

---

## Reference Documentation

For detailed tool descriptions, see:
- `references/tools_reference.md` - Complete tool reference manual

---

## Usage Example

### Example Input

```json
{
  "dataset_description": "Industrial temperature sensor, 1-minute sampling, 120 time steps",
  "series_A": [23.5, 24.1, 23.8, ...],
  "series_B": [23.5, null, 23.9, ...]
}
```

### Example Output

```json
{
  "winner": "A",
  "confidence": 0.85,
  "explanation": "On missing_value dimension, A has missing ratio of 0, B has 0.05, A is better; on noise_level dimension, A has noise_ratio of 0.12, B has 0.18, A is better; comprehensive judgment: A's quality is superior to B."
}
```

---

## Important Notes

1. **Avoid Over-inspection**: Select necessary dimensions based on data characteristics, not all dimensions need assessment
2. **Maintain Parameter Consistency**: Use same parameter settings for A and B
3. **Comprehensive Judgment**: Do not conclude based on single dimension
4. **Interpretability**: explanation should clearly state the basis for judgment
