# TSQualityAgent — Tool Reference Manual

Inspector Agent calls all tools during the ReAct loop. Each tool takes `series_name` ("A" or "B") as the primary parameter, along with optional tuning parameters, and returns results as observations fed back to the model.

---

**Tool Implementation Location**: All the tools elaborated below are implemented in the Python scripts located under the `skills/ts-quality-agent/scripts/` directory:

- **`registry.py`**: The entry point for the agent, exposing `TOOL_REGISTRY` and `TOOL_SCHEMAS`. It registers all tools from other modules.
- **`bad_quality.py`** (Category 1 tools): Contains `missing_ratio`, `noise_profile`, `volatility`, and `range_stats`.
- **`rare_pattern.py`** (Category 2 tools): Contains `zscore_anomaly`, `outlier_density`, `mad_residual_anomaly`, and `contextual_anomaly`.
- **`pattern_structure.py`** (Category 3 tools): Contains `trend_classifier`, `seasonality_detector`, `change_point_detector`, `pattern_consistency_indicators`, `stationarity_test`, `autocorr`, `rolling_amplitude`, and `cycle_amplitude`.

---

## Category 1: Bad Quality

Used to detect data integrity and noise issues, primarily serving `missing_value` and `noise_level` dimensions.

---

### `missing_ratio`

**Calculation Method:** Count NaN values divided by total series length.

**Parameters:** None

**Return Fields:**
| Field | Meaning |
|-------|---------|
| `missing_ratio` | Missing value ratio (0–1). 0 = complete, 1 = all missing |

**Applicable Dimension:** `missing_value`. The most direct metric - simply compare A/B values to draw conclusions.

---

### `noise_profile`

**Calculation Method:** Apply rolling average to the series using a uniform convolution kernel of size `window` to obtain smoothed signal; subtract smoothed signal from original series to get residuals, which serve as noise estimate. Also determine noise type via lag-1 autocorrelation.

**Parameters:**
- `window` (default 5): Rolling window size

**Return Fields:**
| Field | Meaning |
|-------|---------|
| `noise_std` | Standard deviation of residuals, i.e., absolute noise magnitude |
| `signal_std` | Overall standard deviation of original series |
| `noise_ratio` | `noise_std / signal_std`, ratio of noise to total variation, lower is cleaner |
| `noise_type` | `"white"` (random independent noise) or `"red"` (lag-1 autocorrelation > 0.3, structured noise) |

**Limitations:** Rolling mean cannot completely separate periodic components. For strongly periodic series with period < `window`, `noise_std` will be elevated. Recommended to use with `volatility`.

**Applicable Dimension:** `noise_level`, as the primary noise estimation tool.

---

### `volatility`

**Calculation Method:** First compute first-order difference of the series, then slide window over the difference series, taking standard deviation within each window to measure "intensity of step-by-step changes".

**Parameters:**
- `window` (default 20): Sliding window size

**Return Fields:**
| Field | Meaning |
|-------|---------|
| `volatility_mean` | Mean of volatility across all windows |
| `volatility_std` | Standard deviation of volatility across windows |
| `volatility_max` | Maximum window volatility |
| `volatility_trend` | `"increasing"` / `"decreasing"` / `"stable"`, whether volatility changes over time |

**Applicable Dimension:** `noise_level`, as a supplement to `noise_profile`. High `volatility_mean` indicates intense series changes, high `volatility_std` indicates unstable fluctuation.

---

### `range_stats`

**Calculation Method:** Simple statistics of global numerical range of the series.

**Parameters:** None

**Return Fields:**
| Field | Meaning |
|-------|---------|
| `min` | Minimum value (ignoring NaN) |
| `max` | Maximum value (ignoring NaN) |
| `range` | `max - min` |
| `mean` | Mean value |
| `std` | Standard deviation |

**Applicable Dimension:** Auxiliary tool, provides context for other metrics.

---

## Category 2: Rare Pattern

Used to detect anomalies and rare patterns, primarily serving `rare_pattern` dimension.

---

### `zscore_anomaly`

**Calculation Method:** Use Z-score thresholding to detect point anomalies. Points with Z-score exceeding threshold are flagged as anomalies.

**Parameters:**
- `anomaly_threshold` (default 3.0): Number of standard deviations beyond which a point is flagged

**Return Fields:**
| Field | Meaning |
|-------|---------|
| `anomaly_count` | Number of detected anomalies |
| `anomaly_ratio` | Anomaly ratio (count / total length) |
| `anomaly_indices` | Indices of anomaly points |
| `anomaly_values` | Values of anomaly points |
| `threshold_used` | Threshold used for detection |

**Applicable Dimension:** `rare_pattern`, suitable for detecting global point anomalies.

---

### `outlier_density`

**Calculation Method:** Use local density method to detect outliers. Points with significantly lower local density than neighbors are considered outliers.

**Parameters:**
- `k` (default 5): Number of neighbors for density calculation

**Return Fields:**
| Field | Meaning |
|-------|---------|
| `outlier_count` | Number of detected outliers |
| `outlier_ratio` | Outlier ratio |
| `outlier_indices` | Indices of outlier points |
| `outlier_values` | Values of outlier points |

**Applicable Dimension:** `rare_pattern`, suitable for detecting local outliers.

---

### `mad_residual_anomaly`

**Calculation Method:** Use Median Absolute Deviation (MAD) based residual analysis for anomaly detection. More robust to outliers than Z-score method.

**Parameters:**
- `threshold` (default 3.5): MAD-based threshold

**Return Fields:**
| Field | Meaning |
|-------|---------|
| `anomaly_count` | Number of detected anomalies |
| `anomaly_ratio` | Anomaly ratio |
| `anomaly_indices` | Indices of anomaly points |
| `anomaly_values` | Values of anomaly points |
| `threshold_used` | Threshold used for detection |

**Applicable Dimension:** `rare_pattern`, primary tool for global anomaly detection.

---

### `contextual_anomaly`

**Calculation Method:** Detect context-dependent anomalies by comparing each point to its local context window. Anomalies are points that deviate significantly from local pattern.

**Parameters:**
- `context_window` (default 10): Size of context window
- `threshold` (default 3.0): Deviation threshold

**Return Fields:**
| Field | Meaning |
|-------|---------|
| `anomaly_count` | Number of detected anomalies |
| `anomaly_ratio` | Anomaly ratio |
| `anomaly_indices` | Indices of anomaly points |
| `anomaly_values` | Values of anomaly points |
| `threshold_used` | Threshold used for detection |

**Applicable Dimension:** `rare_pattern`, suitable for detecting contextual anomalies in series with local patterns.

---

## Category 3: Pattern Structure

Used to analyze trend, periodicity, amplitude, and consistency features.

---

### `trend_classifier`

**Calculation Method:** Classify trend direction and estimate trend strength via linear regression.

**Parameters:**
- `window` (optional): If provided, use only the last `window` points

**Return Fields:**
| Field | Meaning |
|-------|---------|
| `direction` | `"increasing"` / `"decreasing"` / `"stable"` |
| `slope` | Regression slope |
| `trend_strength` | R² of regression, higher means stronger trend |
| `intercept` | Regression intercept |

**Applicable Dimension:** `trend`, primary tool for trend analysis.

---

### `seasonality_detector`

**Calculation Method:** Detect periodicity through autocorrelation analysis and seasonal decomposition.

**Parameters:**
- `max_period` (default 50): Maximum period to search

**Return Fields:**
| Field | Meaning |
|-------|---------|
| `has_seasonality` | Whether seasonality is detected |
| `period` | Detected period length |
| `seasonal_strength` | Strength of seasonal component |
| `peak_period` | Period with strongest autocorrelation |

**Applicable Dimension:** `frequency`, primary tool for periodicity detection.

---

### `change_point_detector`

**Calculation Method:** Detect structural change points in the series using statistical methods.

**Parameters:**
- `penalty` (default 10): Penalty parameter for change point detection

**Return Fields:**
| Field | Meaning |
|-------|---------|
| `change_points` | List of detected change point indices |
| `change_count` | Number of change points |
| `segments` | List of segment boundaries |

**Applicable Dimension:** `pattern_consistency`, auxiliary tool for structural analysis.

---

### `pattern_consistency_indicators`

**Calculation Method:** Calculate multiple consistency indicators including lumpiness, flat spots, and crossing points.

**Parameters:** None

**Return Fields:**
| Field | Meaning |
|-------|---------|
| `lumpiness` | Measure of series lumpiness |
| `flat_spots` | Number of flat spots |
| `crossing_points` | Number of times series crosses its mean |
| `stability` | Overall stability score |

**Applicable Dimension:** `pattern_consistency`, primary tool for consistency assessment.

---

### `stationarity_test`

**Calculation Method:** Perform statistical test for stationarity (ADF or KPSS test).

**Parameters:**
- `test` (default "adf"): Test type, "adf" or "kpss"

**Return Fields:**
| Field | Meaning |
|-------|---------|
| `is_stationary` | Whether series is stationary |
| `test_statistic` | Test statistic value |
| `p_value` | P-value of test |
| `test_used` | Test method used |

**Applicable Dimension:** Auxiliary tool for trend and consistency analysis.

---

### `autocorr`

**Calculation Method:** Calculate autocorrelation at specified lag.

**Parameters:**
- `lag` (default 1): Lag order

**Return Fields:**
| Field | Meaning |
|-------|---------|
| `autocorr` | Autocorrelation value at specified lag |
| `lag` | Lag order |

**Applicable Dimension:** Auxiliary tool for verifying periodic features. High autocorrelation at `lag=period` indicates strong periodicity.

---

### `rolling_amplitude`

**Calculation Method:** Calculate peak-trough amplitude within sliding window to measure local fluctuation amplitude.

**Parameters:**
- `window` (default 20): Sliding window size

**Return Fields:**
| Field | Meaning |
|-------|---------|
| `mean_amplitude` | Mean amplitude |
| `amplitude_std` | Amplitude standard deviation |
| `amplitude_trend` | Amplitude trend |

**Applicable Dimension:** `amplitude`, for series without obvious periodicity.

---

### `cycle_amplitude`

**Calculation Method:** Identify cycles through peak-trough detection and calculate peak-trough amplitude for each cycle.

**Parameters:** None

**Return Fields:**
| Field | Meaning |
|-------|---------|
| `oscillatory` | **Check this field first**. `False` means series has no obvious oscillation, other fields are unreliable |
| `cycle_count` | Number of detected complete peak-trough pairs |
| `mean_amplitude` | Mean amplitude across cycles |
| `amplitude_cv` | Coefficient of variation of amplitude (std/mean), lower means more consistent amplitude across cycles |
| `amplitude_trend` | `"growing"` / `"shrinking"` / `"stable"`, whether amplitude is modulating |
| `peak_count` | Total number of detected peaks |
| `trough_count` | Total number of detected troughs |

**Applicable Dimension:** `amplitude`, for series with obvious oscillation. Key judgment metric is `amplitude_cv`: low CV + large `mean_amplitude` = high quality. Fall back to `rolling_amplitude` when `oscillatory=False`.

---

## Tool Selection Quick Reference

| Dimension | Primary Tools | Auxiliary Tools |
|-----------|---------------|-----------------|
| `missing_value` | `missing_ratio` | — |
| `noise_level` | `noise_profile`, `volatility` | `range_stats` |
| `rare_pattern` | `mad_residual_anomaly`, `contextual_anomaly` | `zscore_anomaly`, `outlier_density` |
| `trend` | `trend_classifier` | `change_point_detector`, `range_stats`, `stationarity_test` |
| `frequency` | `seasonality_detector` | `autocorr` |
| `amplitude` | `cycle_amplitude` (periodic), `rolling_amplitude` (non-periodic) | `change_point_detector` |
| `pattern_consistency` | `pattern_consistency_indicators` | `stationarity_test`, `change_point_detector` |
