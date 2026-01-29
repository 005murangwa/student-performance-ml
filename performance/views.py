import joblib
import numpy as np
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

# Load the trained model (make sure path is correct)
model = joblib.load('performance/model.pkl')


def _as_float(data, key, errors):
    if key not in data:
        errors[key] = "This field is required."
        return None
    try:
        return float(data[key])
    except (TypeError, ValueError):
        errors[key] = "Must be a number."
        return None


def _as_int(data, key, errors):
    if key not in data:
        errors[key] = "This field is required."
        return None
    try:
        # Reject floats like "2.5"
        value = data[key]
        if isinstance(value, float) and not value.is_integer():
            raise ValueError()
        return int(value)
    except (TypeError, ValueError):
        errors[key] = "Must be an integer."
        return None


def _as_bool(data, key, errors):
    if key not in data:
        errors[key] = "This field is required."
        return None
    value = data[key]
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y"}:
            return True
        if v in {"false", "0", "no", "n"}:
            return False
    errors[key] = "Must be a boolean (true/false)."
    return None


def _validate_inputs(data):
    """
    Validation = type + sane bounds only.
    We don't reject "weird" combos here; instead we classify them into categories
    (tired/overworked/abnormal) via feature engineering.
    """
    errors = {}

    hours_studied = _as_float(data, "hours_studied", errors)
    previous_scores = _as_float(data, "previous_scores", errors)
    sleep_hours = _as_float(data, "sleep_hours", errors)
    sample_papers = _as_int(data, "sample_papers", errors)
    extracurricular = _as_bool(data, "extracurricular", errors)

    # Broad bounds (still reject clearly impossible numbers)
    if hours_studied is not None and not (0 <= hours_studied <= 16):
        errors["hours_studied"] = "Must be between 0 and 16 hours per day."

    if previous_scores is not None and not (0 <= previous_scores <= 100):
        errors["previous_scores"] = "Must be between 0 and 100."

    # Sleeping 16 hours every night is not a realistic "average" for a student.
    if sleep_hours is not None and not (0 <= sleep_hours <= 14):
        errors["sleep_hours"] = "Must be between 0 and 14 hours per night."

    if sample_papers is not None and not (0 <= sample_papers <= 200):
        errors["sample_papers"] = "Must be between 0 and 200."

    if errors:
        return None, errors

    cleaned = {
        "hours_studied": float(hours_studied),
        "previous_scores": float(previous_scores),
        "sleep_hours": float(sleep_hours),
        "sample_papers": int(sample_papers),
        "extracurricular": bool(extracurricular),
    }
    return cleaned, None


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _hard_impossible_checks(cleaned, engineered):
    """
    Some inputs should hard-fail with an error (not just a category),
    because they are effectively impossible as daily averages.
    """
    errors = {}
    h = cleaned["hours_studied"]
    s = cleaned["sleep_hours"]
    day_load = engineered["day_load"]

    # Impossible day allocation (leaves < 4 hours for everything else)
    if day_load > 20:
        errors["hours_studied"] = "Sleep + study exceeds 20 hours/day (impossible daily average)."

    # Very high sleep + high study is not realistic as a stable daily average
    if s >= 14 and h >= 10:
        errors["sleep_hours"] = "Sleeping 14+ hours and studying 10+ hours/day is not realistic."

    # If sleep is extremely high, studying should not be high too
    if s > 12 and h > 8:
        errors["hours_studied"] = "With sleep > 12 hours/night, studying > 8 hours/day is unrealistic."

    return errors


def _engineer_features(cleaned):
    """
    Feature engineering layer (rule-based) to better interpret the raw inputs.
    This does not retrain the model; it produces human-friendly signals.
    """
    h = cleaned["hours_studied"]
    s = cleaned["sleep_hours"]
    p = cleaned["sample_papers"]
    prev = cleaned["previous_scores"]

    # Avoid division by zero while still capturing "no sleep"
    sleep_safe = max(s, 0.5)

    study_sleep_ratio = h / sleep_safe
    day_load = h + s

    # "Effective study" discounts study time if sleep is too low/high.
    sleep_factor = _clamp(s / 8.0, 0.0, 1.15)
    effective_study_hours = h * sleep_factor

    # Simple practice factor (diminishing returns)
    practice_factor = np.log1p(max(p, 0)) / np.log1p(30)  # 0..~1 around 30
    practice_factor = float(_clamp(practice_factor, 0.0, 1.5))

    # A simple readiness score 0..1
    readiness = (
        0.45 * _clamp(prev / 100.0, 0.0, 1.0)
        + 0.35 * _clamp(s / 8.0, 0.0, 1.0)
        + 0.20 * _clamp(practice_factor / 1.0, 0.0, 1.0)
    )
    readiness = float(_clamp(readiness, 0.0, 1.0))

    return {
        "day_load": float(day_load),
        "study_sleep_ratio": float(study_sleep_ratio),
        "effective_study_hours": float(effective_study_hours),
        "practice_factor": float(practice_factor),
        "readiness": float(readiness),
    }


def _categorize_student(cleaned, engineered):
    """
    Categorize into human-friendly buckets + generate flags and tips.
    """
    h = cleaned["hours_studied"]
    s = cleaned["sleep_hours"]
    prev = cleaned["previous_scores"]
    p = cleaned["sample_papers"]

    day_load = engineered["day_load"]
    ratio = engineered["study_sleep_ratio"]

    flags = []
    tips = []

    # Flags
    if s < 4:
        flags.append("very_low_sleep")
        tips.append("Increase sleep closer to 7–9 hours/night for sustainable performance.")
    if s > 11:
        flags.append("very_high_sleep")
        tips.append("If you often sleep 11+ hours, check fatigue/health and keep a consistent routine.")
    if h > 10:
        flags.append("very_high_study")
        tips.append("Studying 10+ hours/day is hard to sustain; add breaks and protect sleep.")
    if day_load > 20:
        flags.append("impossible_daily_load")
        tips.append("Sleep + study exceeds 20 hours/day; numbers likely unrealistic.")
    if ratio > 2.0 and s < 6:
        flags.append("overworked_low_sleep")
        tips.append("Your study-to-sleep balance looks harsh; reduce study hours or increase sleep.")
    if prev < 40 and h < 2 and p < 2:
        flags.append("low_inputs_low_baseline")
        tips.append("Try small consistent habits: +1 hour study/day and a few sample papers each week.")

    # Category selection (ordered from most concerning to best)
    if day_load > 20 or s == 0:
        category = "abnormal"
        label = "Abnormal / unrealistic"
    elif s < 5 or (h >= 8 and s < 6) or "overworked_low_sleep" in flags:
        category = "tired"
        label = "Tired / sleep-deprived"
    elif h >= 10 or day_load > 18:
        category = "overworked"
        label = "Overworked"
    else:
        category = "normal"
        label = "Normal / balanced"

    # A tiny bit of "evaluation" that is separate from the ML model
    if prev >= 75 and category == "normal":
        tips.append("You’re in a strong zone—keep consistency and avoid last-minute cramming.")
    elif prev < 50 and category in {"tired", "overworked"}:
        tips.append("Focus on sleep first; learning efficiency drops hard when tired.")

    return {
        "category": category,
        "label": label,
        "flags": flags,
        "tips": tips[:5],
    }


def _adjust_prediction(model_pred, engineered, category):
    """
    Optional post-processing: adjust the model score slightly based on category/readiness.
    Keeps changes small to avoid 'making up' numbers.
    """
    pred = float(model_pred)
    readiness = engineered["readiness"]  # 0..1

    # Base adjustment from readiness (max ±5)
    adj = (readiness - 0.5) * 10.0

    # Category penalties/bonuses (small)
    if category == "tired":
        adj -= 4.0
    elif category == "overworked":
        adj -= 2.0
    elif category == "abnormal":
        adj -= 6.0
    elif category == "normal":
        adj += 1.0

    final_score = _clamp(pred + adj, 0.0, 100.0)
    return round(final_score, 2), round(adj, 2)


@api_view(['POST'])
def predict_performance(request):
    cleaned, errors = _validate_inputs(request.data)
    if errors:
        return Response(
            {
                "detail": "Input validation failed.",
                "errors": errors,
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    engineered = _engineer_features(cleaned)
    hard_errors = _hard_impossible_checks(cleaned, engineered)
    if hard_errors:
        # User explicitly wants: error + score 0 for impossible inputs
        category = {
            "category": "abnormal",
            "label": "Abnormal / impossible",
            "flags": ["impossible_inputs"],
            "tips": ["Use realistic daily averages (e.g., sleep 6–9 hours, study 1–8 hours)."],
        }
        return Response(
            {
                "detail": "Inputs are unrealistic / impossible.",
                "errors": hard_errors,
                "predicted_performance_index": 0.0,
                "model_prediction": None,
                "adjustment": None,
                "category": category,
                "engineered": engineered,
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    # Convert incoming JSON to features array
    features = np.array([[
        cleaned["hours_studied"],
        cleaned["previous_scores"],
        1 if cleaned["extracurricular"] else 0,
        cleaned["sleep_hours"],
        cleaned["sample_papers"],
    ]])

    # Predict
    model_prediction = float(model.predict(features)[0])

    category_info = _categorize_student(cleaned, engineered)
    final_score, adjustment = _adjust_prediction(model_prediction, engineered, category_info["category"])

    return Response(
        {
            "predicted_performance_index": final_score,
            "model_prediction": round(model_prediction, 2),
            "adjustment": adjustment,
            "category": category_info,
            "engineered": engineered,
        }
    )
