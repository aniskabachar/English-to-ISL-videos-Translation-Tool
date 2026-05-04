# pose_lookup.py — stub only in Phase 1; GlossToKeypoints dict placeholder

# Placeholder for Phase 1
# In future phases, this will map gloss tokens to keypoints/poses

GlossToKeypoints = {
    # Example
    "BOY": "keypoints_for_boy",
    "SCHOOL": "keypoints_for_school",
    "GO": "keypoints_for_go",
    # ...
}

# Function stub
def get_keypoints(gloss_sequence):
    """
    Given a sequence of gloss tokens, return corresponding keypoints.
    For Phase 1, just return placeholder.
    """
    return [GlossToKeypoints.get(gloss, "unknown") for gloss in gloss_sequence]