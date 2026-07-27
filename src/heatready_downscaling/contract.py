"""
ModelAdapter protocol + the v1 QRF implementation of it. Extracted/adapted
from crisisready/heat-risk-data-api's src/downscaling.py (lines ~449-805,
998-1198 as of 2026-07-27, commit 57479e5). See PROVENANCE.md.

Design decision (2026-07-27, see the docstring on ModelAdapter): v1's model
inference logic (what used to be downscaling.predict_downscaled, a free
function taking a raw joblib bundle dict) lives here as QRFModelAdapter, a
concrete class satisfying the ModelAdapter protocol -- not as a separate
"inference" module. This is deliberate: "v1 executes only our code" (the
crowdsourcing program's design principle) means v1's own scoring path
SHOULD run through this same contract a future Rung-C contributor's model
will eventually have to satisfy, rather than a hardcoded free function with
the interface bolted on as an afterthought. score.score_band depends on
ModelAdapter, never on QRFModelAdapter directly.

RESOLVED 2026-07-27 (Phase 3, see FrozenPredictionAdapter below): the open
question this docstring used to raise -- QRFModelAdapter.load() needs
private S3 credentials no external contributor or CI job has, so how does
Rung A/B scoring get model predictions at all? -- is answered by NOT
distributing the model artifact itself. Instead, the maintainer freezes
QRFModelAdapter's own per-row predictions (delta_c, ci95_c, confidence,
etc.) into a new snapshot partition (heatready_downscaling.snapshot.
write_predictions_partition), and FrozenPredictionAdapter reads that
partition back as a lookup table satisfying the exact same ModelAdapter
protocol. This removes AWS access AND the sklearn/joblib pickle-compat
requirement from the entire contributor/CI scoring path -- reproduction
becomes a byte-exact lookup, not a re-run of live inference, which is
actually a STRONGER reproducibility guarantee than shipping the pickle
would have been (no risk of a contributor's local sklearn/joblib/
quantile-forest version silently producing different numbers).
"""

import logging
from typing import Protocol

logger = logging.getLogger(__name__)

_QUANTILES = (0.025, 0.5, 0.975)  # QRF's own 95% interval, before conformal scaling
_CONFIDENCE_HIGH_CI95_C = 1.0
_CONFIDENCE_MEDIUM_CI95_C = 2.5


class ModelAdapter(Protocol):
    """Structural interface a v1 QRF bundle (QRFModelAdapter below) AND a
    future Rung-C contributor model must both satisfy. score.score_band and
    any replay/promotion tooling depend on this protocol, never on a
    concrete class -- so a Rung-C model, once that rung opens, is a drop-in
    for QRFModelAdapter with no change to score.py."""

    model_version: str

    def predict(
        self,
        rows: list[dict],
        target: str,
        extra_zone_gate: dict[str, dict[str, bool]] | None = None,
        bias_correction: dict[str, dict[str, float]] | None = None,
    ) -> list[dict]:
        """Predict downscaled delta_tmax_c/delta_tmin_c (target picks
        which) plus the full uncertainty/provenance block for a batch of
        polygon-day/station-day covariate rows (same shape
        features.build_feature_matrix expects).

        Returns one dict per input row, in the same order, with keys:
        delta_c, ci95_c, confidence, applied, out_of_distribution,
        covariates_missing, cv_gate_passed, model_version -- see
        QRFModelAdapter.predict's own docstring for the exact semantics of
        each, which every implementation of this protocol must preserve."""
        ...


def validate_feature_order(feature_order) -> None:
    """Raise if `feature_order` (as read from a model's own metadata.json)
    does not match features.FEATURE_ORDER exactly. build_feature_matrix's X
    columns are positional, so a model trained against a different column
    order would silently score every feature against the wrong column -- no
    shape mismatch, no exception, just quietly wrong predictions. This is
    the stated precondition for Rung C: a contributor's model must be
    provably trained against the exact feature contract this module scores
    against, or it must not load at all."""
    from heatready_downscaling.features import FEATURE_ORDER

    live_feature_order = tuple(feature_order)
    if live_feature_order != FEATURE_ORDER:
        raise ValueError(
            "model metadata.json feature_order does not match "
            "heatready_downscaling.features.FEATURE_ORDER -- refusing to load a model "
            "whose training-time column order does not match this package's current "
            f"build_feature_matrix. metadata: {live_feature_order!r} != code: {FEATURE_ORDER!r}",
        )


def feature_importance_weights(model) -> "np.ndarray":
    """Normalize an RF/QRF model's feature_importances_ to sum to 1 -- the
    AOA weighting vector (Meyer & Pebesma 2021)."""
    import numpy as np
    importances = np.asarray(model.feature_importances_, dtype=float)
    total = importances.sum()
    if total <= 0:
        return np.ones_like(importances) / len(importances)
    return importances / total


def aoa_dissimilarity(
    X_query: "np.ndarray", X_train: "np.ndarray", weights: "np.ndarray",
    train_mean: "np.ndarray", train_std: "np.ndarray",
) -> "np.ndarray":
    """
    AOA-style (Meyer & Pebesma 2021) dissimilarity index: feature-importance
    -weighted Euclidean distance, in z-score-standardized feature space, from
    each query point to its nearest training point. A decided simplification
    of the published method -- a weighted distance to the nearest training
    sample, normalized (by the caller, against the CV fold-average distance)
    rather than the full CAST-package percentile-based AOA threshold. Shared
    by training (computing the CV fold-average distance) and inference
    (scoring a new row against the shipped model's stored training sample)
    so both sides compute the identical metric.

    train_mean/train_std must come from the SAME reference distribution
    (either one CV fold's training split, or the final model's full training
    set) as X_train itself -- mixing scales from different fits would make
    the resulting distances incomparable.
    """
    from sklearn.neighbors import NearestNeighbors
    import numpy as np

    train_std_safe = np.where(train_std > 0, train_std, 1.0)
    scaled_train = ((X_train - train_mean) / train_std_safe) * np.sqrt(weights)
    scaled_query = ((X_query - train_mean) / train_std_safe) * np.sqrt(weights)
    nn = NearestNeighbors(n_neighbors=1).fit(scaled_train)
    dist, _ = nn.kneighbors(scaled_query)
    return dist[:, 0]


def _confidence_class(ci95_c: float | None, out_of_distribution: bool) -> str:
    if ci95_c is None:
        return "low"
    if out_of_distribution:
        return "low"
    if ci95_c <= _CONFIDENCE_HIGH_CI95_C:
        return "high"
    if ci95_c <= _CONFIDENCE_MEDIUM_CI95_C:
        return "medium"
    return "low"


def _not_applied_result(
    model_version: str,
    covariates_missing: list[str] | None = None,
    cv_gate_passed: bool | None = None,
    out_of_distribution: bool | None = None,
) -> dict:
    """Shared shape for QRFModelAdapter.predict's two 'not applied' cases
    (incomplete covariates vs. a failed CV gate) -- keeps the two branches
    from silently drifting apart on which keys they set."""
    return {
        "delta_c": None, "ci95_c": None, "confidence": "low", "applied": False,
        "out_of_distribution": out_of_distribution,
        "covariates_missing": covariates_missing or [],
        "cv_gate_passed": cv_gate_passed, "model_version": model_version,
    }


def derive_zones_passing_cv_gate(metadata: dict) -> dict[str, dict[str, bool]]:
    """
    {target: {zone: qrf_beats_grid}} derived from the nested leave-region-out
    CV report already present in metadata.json
    (cv.leave_region_out[target].by_zone[zone].qrf_beats_grid).
    QRFModelAdapter.predict must not apply a correction in a zone that
    fails its own beat-the-grid gate.

    Deliberately derived at load time from the existing nested structure,
    NOT read from a new flat metadata field -- this guarantees scoring
    reads the exact same qrf_beats_grid values training reported, no drift
    risk, the same reason features.build_feature_matrix is a single shared
    function rather than independent implementations.

    A zone absent from the CV report (never sampled by the training set, or
    dropped for too few points) is treated as NOT passing -- fail closed.
    This is the conservative default for a correctness-critical design whose
    own principle is "a wrong-but-confident correction is worse than an
    honest fallback to the raw grid value."
    """
    leave_region_out = (metadata.get("cv") or {}).get("leave_region_out") or {}
    if not leave_region_out:
        logger.error(
            "Model metadata for '%s' has no cv.leave_region_out section -- "
            "zones_passing_cv_gate will be empty and every zone will fail "
            "closed (fall back to grid). This is either a genuinely CV-free "
            "model artifact or a metadata schema change QRFModelAdapter.predict "
            "was not updated for -- investigate before assuming the model "
            "just doesn't apply anywhere.",
            metadata.get("model_version", "<unknown>"),
        )
    out: dict[str, dict[str, bool]] = {}
    for target in ("tmax", "tmin"):
        by_zone = (leave_region_out.get(target) or {}).get("by_zone") or {}
        out[target] = {zone: bool(info.get("qrf_beats_grid")) for zone, info in by_zone.items()}
    return out


class QRFModelAdapter:
    """v1's only concrete ModelAdapter -- wraps a joblib bundle
    {model_tmax, model_tmin, metadata, zones_passing_cv_gate,
    aoa_train_features_{target}, aoa_feature_weights_{target},
    aoa_train_mean_{target}, aoa_train_std_{target}} produced by
    scripts/train_downscaling.py."""

    def __init__(self, bundle: dict):
        self._bundle = bundle
        self.model_version = bundle["metadata"]["model_version"]

    @classmethod
    def load(cls, model_version: str, bucket: str | None = None) -> "QRFModelAdapter":
        """Load model.joblib + metadata.json from
        s3://{bucket}/downscaling/models/{model_version}/. bucket defaults
        to the VULNERABILITY_DATA_BUCKET env var. See this module's own
        docstring for the open question about external-contributor access
        to this artifact -- this method's transport is the maintainer's
        own (private-bucket) path, not necessarily a public one."""
        import os
        import tempfile

        import boto3
        import joblib

        bucket = bucket or os.environ["VULNERABILITY_DATA_BUCKET"]
        client = boto3.client("s3")
        prefix = f"downscaling/models/{model_version}/"

        # Stream model.joblib to a /tmp file and load from the path, rather
        # than `.read()`-ing the whole object into memory -- avoids holding
        # two full in-memory copies of a potentially multi-GB artifact
        # simultaneously.
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            client.download_file(bucket, f"{prefix}model.joblib", tmp_path)
            bundle = joblib.load(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                logger.warning("Could not remove temp model file %s", tmp_path, exc_info=True)

        import json
        metadata = json.loads(client.get_object(Bucket=bucket, Key=f"{prefix}metadata.json")["Body"].read())

        validate_feature_order(metadata["feature_order"])

        bundle["metadata"] = metadata
        bundle["zones_passing_cv_gate"] = derive_zones_passing_cv_gate(metadata)
        logger.info("Loaded downscaling model '%s' from s3://%s/%s", model_version, bucket, prefix)
        return cls(bundle)

    def predict(
        self,
        rows: list[dict],
        target: str,
        extra_zone_gate: dict[str, dict[str, bool]] | None = None,
        bias_correction: dict[str, dict[str, float]] | None = None,
    ) -> list[dict]:
        """
        See ModelAdapter.predict for the return shape. Full semantics,
        ported unchanged from downscaling.predict_downscaled:

        extra_zone_gate: an ADDITIONAL {target: {zone: bool}} restriction,
        ANDed with the model's own zones_passing_cv_gate. None (the
        default) means "no additional restriction beyond the model's
        built-in ERA5-band gate" -- correct for the ERA5 band itself.
        Callers scoring a DIFFERENT base distribution (lag-fill, forecast
        lead=1) must always pass a real (possibly all-empty, i.e.
        fail-closed) dict here.

        bias_correction: an ADDITIONAL {target: {zone: float}} constant
        ADDED to delta_c after the model predicts it (MOS-style
        recalibration). None (the default) applies no correction.

        The CV gate is enforced independently per call (i.e. per target).
        Never derives the downscaled T2m or re-runs heat_calcs itself --
        that join and any HI/WBGT/UTCI re-derivation is the caller's job.
        """
        import numpy as np

        from heatready_downscaling.features import build_feature_matrix

        model_bundle = self._bundle
        model = model_bundle["model_tmax"] if target == "tmax" else model_bundle["model_tmin"]
        metadata = model_bundle["metadata"]
        model_version = metadata["model_version"]
        q95_key = "conformal_q95_by_zone" if target == "tmax" else "conformal_q95_by_zone_tmin"
        q95_by_zone = metadata.get(q95_key, {})
        ood_threshold = metadata.get("ood_aoa_threshold")

        X, complete_mask, missing_by_row = build_feature_matrix(rows, target)
        results: list[dict] = []

        gate_for_target = model_bundle.get("zones_passing_cv_gate", {}).get(target, {})
        extra_gate_for_target = extra_zone_gate.get(target, {}) if extra_zone_gate is not None else None
        bias_correction_for_target = bias_correction.get(target, {}) if bias_correction is not None else {}
        gate_passed = [
            complete_mask[i]
            and gate_for_target.get(r.get("climate_zone"), False)
            and (extra_gate_for_target is None or extra_gate_for_target.get(r.get("climate_zone"), False))
            for i, r in enumerate(rows)
        ]

        complete_idx = [i for i, ok in enumerate(complete_mask) if ok]
        predict_idx = [i for i in complete_idx if gate_passed[i]]
        preds = np.zeros((len(rows), 3))
        di = np.full(len(rows), np.nan)
        if complete_idx:
            di[complete_idx] = aoa_dissimilarity(
                X[complete_idx], model_bundle[f"aoa_train_features_{target}"], model_bundle[f"aoa_feature_weights_{target}"],
                model_bundle[f"aoa_train_mean_{target}"], model_bundle[f"aoa_train_std_{target}"],
            )
        if predict_idx:
            preds[predict_idx] = model.predict(X[predict_idx], quantiles=list(_QUANTILES))

        for i, r in enumerate(rows):
            if not complete_mask[i]:
                results.append(_not_applied_result(model_version, covariates_missing=missing_by_row[i]))
                continue

            out_of_distribution = bool(ood_threshold and di[i] > ood_threshold)
            if not gate_passed[i]:
                results.append(_not_applied_result(
                    model_version, cv_gate_passed=False, out_of_distribution=out_of_distribution,
                ))
                continue

            lo, median, hi = preds[i]
            zone = r.get("climate_zone")
            q95 = q95_by_zone.get(zone, q95_by_zone.get("_default", 1.0))
            raw_half_width = (hi - lo) / 2.0
            ood_inflation = (di[i] / ood_threshold) if (ood_threshold and out_of_distribution) else 1.0
            ci95_c = raw_half_width * q95 * ood_inflation
            delta_c = float(median) + bias_correction_for_target.get(zone, 0.0)

            results.append({
                "delta_c": delta_c,
                "ci95_c": float(ci95_c),
                "confidence": _confidence_class(ci95_c, out_of_distribution),
                "applied": True,
                "out_of_distribution": out_of_distribution,
                "covariates_missing": [],
                "cv_gate_passed": True,
                "model_version": model_version,
            })

        return results


class FrozenPredictionAdapter:
    """A ModelAdapter backed by a snapshot's frozen predictions partition
    (heatready_downscaling.snapshot.write_predictions_partition/
    read_predictions_partitions) instead of a live joblib model -- see this
    module's own docstring for why this exists.

    The frozen data represents QRFModelAdapter.predict's output called with
    extra_zone_gate=None and bias_correction=None -- i.e. the base model's
    raw result, gated ONLY by its own trained-on-ERA5 zones_passing_cv_gate.
    predict() below re-applies extra_zone_gate/bias_correction at call time
    against that frozen baseline, replicating QRFModelAdapter.predict's
    exact AND-gate/addition semantics:

    - A row frozen as NOT applied (incomplete covariates, or failed the
      base ERA5 CV gate) stays not-applied regardless of extra_zone_gate --
      extra_zone_gate can only ever ADD a restriction, never rescue a row
      the base gate already failed, exactly like the live model.
    - A row frozen as applied, but whose zone extra_zone_gate marks False,
      becomes not-applied (cv_gate_passed=False) -- the SAME transition
      QRFModelAdapter.predict makes when the AND of its two gates fails.
    - A row frozen as applied AND passing extra_zone_gate (or
      extra_zone_gate is None) keeps its frozen delta_c/ci95_c/confidence,
      with bias_correction_for_target.get(zone, 0.0) ADDED to delta_c --
      ci95_c is untouched by bias_correction, matching the live model
      (ci95_c is derived from conformal quantile width alone, never from
      the bias correction)."""

    def __init__(self, model_version: str, predictions_by_key: dict[tuple, dict]):
        self.model_version = model_version
        self._predictions = predictions_by_key

    @classmethod
    def from_snapshot(
        cls, snapshot_dir: str, model_version: str, band: str, months: list[str] | None = None,
    ) -> "FrozenPredictionAdapter":
        """Load predictions/model={model_version}/band={band}/month=*/
        part-0.parquet from `snapshot_dir` and index by
        (station_id, date_iso, target)."""
        from heatready_downscaling.snapshot import read_predictions_partitions

        rows = read_predictions_partitions(snapshot_dir, model_version, band, months=months)
        predictions_by_key: dict[tuple, dict] = {}
        for r in rows:
            d = r["date"]
            date_iso = d.isoformat() if hasattr(d, "isoformat") else str(d)
            predictions_by_key[(r["station_id"], date_iso, r["target"])] = r
        return cls(model_version, predictions_by_key)

    def predict(
        self,
        rows: list[dict],
        target: str,
        extra_zone_gate: dict[str, dict[str, bool]] | None = None,
        bias_correction: dict[str, dict[str, float]] | None = None,
    ) -> list[dict]:
        """See ModelAdapter.predict for the return shape; see this class's
        own docstring for the exact re-gating/bias-correction semantics."""
        extra_gate_for_target = extra_zone_gate.get(target, {}) if extra_zone_gate is not None else None
        bias_correction_for_target = bias_correction.get(target, {}) if bias_correction is not None else {}

        results: list[dict] = []
        for r in rows:
            d = r.get("date")
            date_iso = d.isoformat() if hasattr(d, "isoformat") else str(d)
            zone = r.get("climate_zone")
            frozen = self._predictions.get((r.get("station_id"), date_iso, target))

            if frozen is None:
                # Not in the frozen partition at all -- a real data-shape
                # mismatch between the paired rows and the predictions
                # partition being scored against, not an expected "gate
                # failed" case. Fail closed (not applied) rather than
                # raising, matching this module's general "a wrong-but-
                # confident correction is worse than an honest fallback"
                # posture, but this should never happen for a
                # correctly-paired (snapshot_version, model_version, band)
                # combination -- investigate if it does.
                logger.warning(
                    "No frozen prediction for station_id=%s date=%s target=%s model_version=%s -- "
                    "check the predictions partition actually covers this station-day.",
                    r.get("station_id"), date_iso, target, self.model_version,
                )
                results.append(_not_applied_result(self.model_version))
                continue

            if not frozen["applied"]:
                results.append(_not_applied_result(
                    self.model_version,
                    covariates_missing=frozen["covariates_missing"],
                    cv_gate_passed=frozen["cv_gate_passed"],
                    out_of_distribution=frozen["out_of_distribution"],
                ))
                continue

            if extra_gate_for_target is not None and not extra_gate_for_target.get(zone, False):
                results.append(_not_applied_result(
                    self.model_version, cv_gate_passed=False, out_of_distribution=frozen["out_of_distribution"],
                ))
                continue

            results.append({
                "delta_c": frozen["delta_c"] + bias_correction_for_target.get(zone, 0.0),
                "ci95_c": frozen["ci95_c"],
                "confidence": frozen["confidence"],
                "applied": True,
                "out_of_distribution": frozen["out_of_distribution"],
                "covariates_missing": [],
                "cv_gate_passed": True,
                "model_version": self.model_version,
            })

        return results
