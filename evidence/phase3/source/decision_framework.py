"""Illustrative, nonbinding gate with mandatory human authorization."""

def recommend(ci, n_cases, config):
    if config.get("safety_veto", False):
        status, reason = "recommend_rejection", "Prespecified safety veto"
    elif n_cases < int(config["minimum_cases"]):
        status, reason = "recommend_additional_evaluation", "Insufficient independent cases"
    elif ci["lower"] > float(config["rejection_tolerance"]):
        status, reason = "recommend_rejection", "Evidence of practically important log-loss regression"
    elif ci["upper"] < -float(config["minimum_logloss_improvement"]):
        status, reason = "recommend_approval", "Confidence bound supports prespecified practical improvement"
    else:
        status, reason = "recommend_additional_evaluation", "Uncertainty does not resolve improvement/regression criteria"
    return {"status":status,"reason":reason,"final_human_approval_required":True}
